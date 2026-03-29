import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from flax import linen as nn
from typing import Dict, Any, Sequence, Tuple

global_mesh = None
axis_to_dim = dict()

def set_global_mesh(hsdp_dim: int = 8):
    global global_mesh
    global axis_to_dim
    hsdp_dim = min(hsdp_dim, jax.process_count() * jax.local_device_count())
    mesh_shape = (jax.process_count() * jax.local_device_count() // hsdp_dim, hsdp_dim)
    axis_names = ['data', 'fsdp']
    axis_to_dim['fsdp'] = hsdp_dim
    axis_to_dim['data'] = jax.process_count() * jax.local_device_count() // hsdp_dim
    devices = mesh_utils.create_device_mesh(mesh_shape, allow_split_physical_axes=True)
    global_mesh = Mesh(devices, axis_names=axis_names)

def get_global_mesh():
    global global_mesh
    assert global_mesh is not None, "Global mesh is not set"
    return global_mesh

def axis_dim(axis: str):
    global axis_to_dim
    return axis_to_dim[axis]

def get_spec(path_name, tensor_leaf, axis_tuple=('fsdp',)):
    # when shard_in: prioritize sharding in first dim
    dim_val = 1
    for axis in axis_tuple:
        dim_val *= axis_dim(axis)
    n_dim = tensor_leaf.ndim
    if n_dim == 0:
        return P()
    for i in range(n_dim - 1, -1, -1):
        if tensor_leaf.shape[i] % dim_val == 0:
            return P(*([None] * i + [axis_tuple]))
    if tensor_leaf.shape[0] % dim_val == 0:
        return P(axis_tuple)
    elif tensor_leaf.shape[-1] % dim_val == 0:
        return P(*([None] * (n_dim - 1) + [axis_tuple]))
    elif len(tensor_leaf.shape) >= 2 and tensor_leaf.shape[-2] % dim_val == 0:
        return P(*([None] * (n_dim - 2) + [axis_tuple] + [None]))
    else:
        if len(axis_tuple) >= 2:
            sub_spec = get_spec(tensor_leaf, axis_tuple=axis_tuple[:-1])
            sub_spec_2 = get_spec(tensor_leaf, axis_tuple=axis_tuple[1:])
            if sub_spec != P():
                return sub_spec
            elif sub_spec_2 != P():
                return sub_spec_2
    return P()


_STATIC_TYPES = (bool, int, float, str)

def split_static_dynamic(dummy_input: dict):
    """Split dummy_input into (dynamic_input, static_input).

    Rules:
      - bool/int/float/str -> static
      - everything else (arrays, ShapeDtypeStruct, numpy, etc.) -> dynamic
    """
    dynamic_input = {}
    static_input = {}
    for k, v in dummy_input.items():
        if isinstance(v, _STATIC_TYPES):
            static_input[k] = v
        else:
            dynamic_input[k] = v
    return dynamic_input, static_input

def prepare_rngs(rng, all_keys: Sequence[str]):
    total_keys_needed = len(all_keys)
    splitted_keys = jax.random.split(rng, total_keys_needed)
    rngs = {}
    for i, key_name in enumerate(all_keys):
        rngs[key_name] = splitted_keys[i]
    return rngs

def ddp_shard():
    return NamedSharding(get_global_mesh(), P(('data', 'fsdp')))

def data_shard():
    return NamedSharding(get_global_mesh(), P(('data',)))

def enforce_ddp(x):
    def try_ddp_shard(x):
        if x.shape[0] % jax.device_count() != 0:
            return x
        else:
            return jax.lax.with_sharding_constraint(x, ddp_shard())
    return jax.tree.map(try_ddp_shard, x)

def init_state_from_dummy_input(
    model,
    optimizer,
    TrainState,
    rng, 
    dummy_input: dict,
    rng_keys_extra: Sequence[str] = [], # rng_keys not including params
    ema_decay=0.999,
    **_unused_kwargs,
):
    """
    Initialize TrainState with:
      1) eval_shape on init to get abstract_state (no big allocations)
      2) build state_shardings from abstract_state
      3) jit(init_fn, out_shardings=state_shardings) to materialize sharded state directly

    Requirements:
      - dummy_input keys match model.init kwargs (except rngs)
      - make_state_shardings must be provided if you want sharded init
      - mesh must be provided if using NamedSharding-based shardings
    """
    dynamic_input, static_input = split_static_dynamic(dummy_input)
    rngs = prepare_rngs(rng, ['params'] + rng_keys_extra)
    if isinstance(ema_decay, (list, tuple)):
        if len(ema_decay) != 1:
            raise ValueError(f"Expected a single ema_decay value, got {ema_decay}")
        ema_decay = float(ema_decay[0])
    else:
        ema_decay = float(ema_decay)

    def init_state_fn(rngs_inner, dynamic_inputs_inner):
        variables = model.init(rngs_inner, **dynamic_inputs_inner, **static_input)
        params = variables["params"]
        opt_state = optimizer.init(params)
        ema_params = jax.tree.map(lambda x: x, params)
        return TrainState(
            step=jnp.array(0, dtype=jnp.int32),
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
            opt_state=opt_state,
            ema_params=ema_params,
            ema_decay=ema_decay,
        )

    abstract_state = jax.eval_shape(init_state_fn, rngs, dynamic_input)
    mesh = get_global_mesh()
    def map_fn(path, value):
        path_str = "/".join([str(p) for p in path])
        if hasattr(value, "shape") and value.ndim > 0:
            axis = ('fsdp',)
            return NamedSharding(mesh, get_spec(path_str, value, axis_tuple=axis))
        else:
            return NamedSharding(mesh, P())

    state_shardings = jax.tree_util.tree_map_with_path(map_fn, abstract_state)

    init_compiled = jax.jit(init_state_fn, out_shardings=state_shardings)
    state = init_compiled(rngs, dynamic_input)

    return state

def map_to_sharding(params):
    """Return a function that reshards inputs to match `params` sharding."""
    target_shardings = jax.tree.map(lambda x: x.sharding, params)

    def reshard_fn(other_params):
        return other_params

    return jax.jit(reshard_fn, out_shardings=target_shardings)
    
def init_model_distributed(
    model: nn.Module,
    dummy_input: Dict[str, Any],
    rng: jax.Array | None = None,
    rng_keys_extra: Sequence[str] = [],
) -> Tuple[Any, Mesh]:
    """
    Args:
        model: Flax Module; 
        dummy_input: input dict passing to the model; 
        master_rng: main seed; 
        rng_keys: extra rng keys needed, e.g. ['dropout', 'gating']
        mode: 'hsdp' or 'ddp'
        
    Returns:
        params: initialized parameters. 
    """
    
    rng = jax.random.PRNGKey(0) if rng is None else rng
    rngs = prepare_rngs(rng, ['params'] + rng_keys_extra)
    mesh = get_global_mesh()
    dynamic_input, static_input = split_static_dynamic(dummy_input)

    def _init_wrapper(rngs_inner, inputs_inner):
        return model.init(rngs_inner, **inputs_inner, **static_input)

    abstract_variables = jax.eval_shape(
        _init_wrapper, rngs, dynamic_input
    )
    sharding_spec_tree = jax.tree_util.tree_map_with_path(
        lambda path, x: NamedSharding(mesh, get_spec(path, x, axis_tuple=('fsdp',))),
        abstract_variables
    )
    def _jit_init_fn(rngs_inner, inputs_inner):
        return model.init(rngs_inner, **inputs_inner, **static_input)
    
    jit_init_fn = jax.jit(_jit_init_fn, out_shardings=sharding_spec_tree)

    variables = jit_init_fn(rngs, dynamic_input)
    
    return variables


def merge_data(data, use_ddp=False):
    '''
    Args:
        data: pytree; data on current process; 
        use_ddp: bool; whether to use ddp to merge data;
            When true: ddp; shard across devices; 
            When false: data_shard(); shard across processes; 
    Returns:
        data: pytree; data on all devices
    
    '''

    fsdp_mesh = get_global_mesh()
    def auto_shard(x):
        sharding = [('data', 'fsdp')] 
        sharding = NamedSharding(fsdp_mesh, P(*sharding))
        return jax.make_array_from_process_local_data(
            sharding,
            x,
        )
    ddp_sharded = jax.tree.map(auto_shard, data)
    if not use_ddp:
        return jax.tree.map(lambda x : jax.device_put(x, data_shard()), ddp_sharded)
    else:
        return ddp_sharded
    
def pad_and_merge(data, local_bsz, use_ddp=False):
    '''
    Args:
        data: pytree; data on current process; 
        local_bsz: int; desired local batch size (e.g. 32)
    Returns:
        data_merged: pytree; global data (JAX Array)
        mask_merged: jax.Array; global mask (1=valid, 0=padding)
    '''
    leaves = jax.tree.leaves(data)
    if not leaves:
        raise ValueError("Data is empty")
    current_len = leaves[0].shape[0]
    
    # 2. prepare padding and mask (on CPU/Numpy side)
    pad_len = local_bsz - current_len
    assert pad_len >= 0, f"local_bsz: {local_bsz} is less than current_len: {current_len}"
    local_mask = jnp.concatenate([
        jnp.ones(current_len, dtype=jnp.int32),
        jnp.zeros(pad_len, dtype=jnp.int32)
    ])
    
    def pad_leaf(x):
        x = jnp.asarray(x)
        if pad_len == 0:
            return x
        pad_width = [(0, pad_len)] + [(0, 0)] * (x.ndim - 1)
        return jnp.pad(x, pad_width, mode='constant', constant_values=0)
    
    local_data_padded = jax.tree_map(pad_leaf, data)
    return merge_data(local_data_padded, use_ddp), merge_data(local_mask, use_ddp)
