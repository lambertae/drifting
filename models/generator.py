import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
import numpy as np
from typing import Optional, Dict, Any
from utils.hsdp_util import enforce_ddp
from utils.misc import ddp_rand_func
import math
from utils.env import HF_REPO_ID, HF_ROOT

# -----------------------------------------------------------------------------
# 1. Utils & Base Modules (Fixed Precision & Init)
# -----------------------------------------------------------------------------

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Sinusoidal positional encoding for a 1-D grid.

    Args:
        embed_dim: embedding dimension (must be even).
        pos: grid positions, any shape (flattened internally).

    Returns:
        np.ndarray of shape ``(len(pos), embed_dim)``.
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega 
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """2-D sinusoidal positional encoding.

    Returns:
        np.ndarray of shape ``(grid_size * grid_size, embed_dim)``.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h) # w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    embed_dim_half = embed_dim // 2
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim_half, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim_half, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)

def sincos_init(embed_dim, num_patches):
    """Flax parameter initializer returning a 2-D sincos positional embedding of shape ``(1, num_patches, embed_dim)``."""
    def init_fn(key, shape, dtype=jnp.float32):
        grid_size = int(np.sqrt(num_patches))
        pe = get_2d_sincos_pos_embed(embed_dim, grid_size)
        return jnp.asarray(pe, dtype=dtype)[None, :, :] # [1, T, D]
    return init_fn

class TorchLinear(nn.Module):
    """Linear layer strictly matching PyTorch defaults."""
    features: int
    bias: bool = True
    weight_init: str = "xavier_uniform"
    bias_init: str = "zeros"
    # Always explicit (no None): either fp32 or bf16.
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        w_init_fn = {
            "xavier_uniform": nn.initializers.xavier_uniform(),
            "zeros": nn.initializers.zeros,
            "normal": nn.initializers.normal(stddev=0.02),
        }.get(self.weight_init, nn.initializers.xavier_uniform())
        
        b_init_fn = nn.initializers.zeros if self.bias_init == "zeros" else nn.initializers.constant(0.0)

        x = x.astype(self.dtype)
        return nn.Dense(
            self.features,
            use_bias=self.bias,
            kernel_init=w_init_fn,
            bias_init=b_init_fn,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)

class RMSNorm(nn.Module):
    dim: int # Kept for arg compatibility
    eps: float = 1e-6
    elementwise_affine: bool = True

    @nn.compact
    def __call__(self, x):
        input_dtype = x.dtype
        # [Precision Fix]: Cast to float32 for variance calculation match PyTorch behavior
        var_x = x.astype(jnp.float32)
        var = jnp.mean(jnp.square(var_x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(var + self.eps)
        
        if self.elementwise_affine:
            scale = self.param('weight', nn.initializers.ones, (x.shape[-1],))
            normed = normed * scale
        return normed.astype(input_dtype)

def modulate(x, shift, scale):
    """AdaLN modulation: ``x * (1 + scale) + shift``, broadcasting over the token dimension."""
    return x * (1 + jnp.expand_dims(scale, axis=1)) + jnp.expand_dims(shift, axis=1)

def apply_rope(q, k, dtype=jnp.float32):
    """
    Apply Rotary Positional Embedding to q and k.
    q, k: [B, N, H, D]
    """
    B, N, H, D = q.shape
    half_dim = D // 2
    freqs = (1.0 / (10000 ** (jnp.arange(0, half_dim) / half_dim))).astype(dtype)
    t = jnp.arange(N, dtype=dtype)
    freqs = jnp.outer(t, freqs) # [N, D/2]
    emb = jnp.concatenate([freqs, freqs], axis=-1)

    cos = jnp.cos(emb)[None, :, None, :]
    sin = jnp.sin(emb)[None, :, None, :]
    def rotate_half(x):
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        return jnp.concatenate([-x2, x1], axis=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SwiGLUFFN(nn.Module):
    hidden_size: int
    intermediate_size: int
    # Always explicit (no None): either fp32 or bf16.
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        w1 = TorchLinear(self.intermediate_size, bias=True, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        w3 = TorchLinear(self.intermediate_size, bias=True, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        out = nn.silu(w1) * w3
        return TorchLinear(self.hidden_size, bias=True, dtype=self.dtype, param_dtype=self.param_dtype)(out)

# -----------------------------------------------------------------------------
# 2. Core Blocks (Fixed Attention Norm & MLP)
# -----------------------------------------------------------------------------

class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    qk_norm: bool = False
    use_rmsnorm: bool = False
    use_rope: bool = False
    attn_drop: float = 0.
    proj_drop: float = 0.
    attn_fp32: bool = True  
    # Always explicit (no None): either fp32 or bf16.
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, deterministic=True, return_qk=False):
        """
        Args:
            x: [B, N, C]
            deterministic: disable dropout when True
            return_qk: if True, also return (q, k) each [B, N, num_heads, head_dim]
                after norm/rope but before scaling
        Returns:
            x: [B, N, C]
            qk: tuple (q, k) each [B, N, num_heads, head_dim] if return_qk, else None
        """
        B, N, C = x.shape
        head_dim = self.dim // self.num_heads

        qkv = TorchLinear(self.dim * 3, bias=self.qkv_bias, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim)
        q, k, v = qkv[:, :, 0, :, :], qkv[:, :, 1, :, :], qkv[:, :, 2, :, :]

        # [Logic Fix]: Correct instantiation of Norms
        if self.qk_norm:
            if self.use_rmsnorm:
                q = RMSNorm(head_dim, name='q_norm')(q)
                k = RMSNorm(head_dim, name='k_norm')(k)
            else:
                # LayerNorm args: epsilon, then use_scale/bias. Don't pass dim as pos arg.
                q = nn.LayerNorm(epsilon=1e-6, use_scale=True, use_bias=True, name='q_norm')(q)
                k = nn.LayerNorm(epsilon=1e-6, use_scale=True, use_bias=True, name='k_norm')(k)
        if self.use_rope:
            rope_dtype = jnp.float32 if self.attn_fp32 else self.dtype
            q, k = apply_rope(q, k, dtype=rope_dtype)

        # Capture q, k after norm/rope but before scaling
        qk = (q, k) if return_qk else None  # each [B, N, num_heads, head_dim]

        if self.attn_fp32:
            q = q.astype(jnp.float32) * (head_dim ** -0.5)
            k = k.astype(jnp.float32)
            v = v.astype(jnp.float32)
        else:
            q = q.astype(self.dtype) * (head_dim ** -0.5)
            k = k.astype(self.dtype)
            v = v.astype(self.dtype)

        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        attn_logits = jnp.matmul(q, jnp.swapaxes(k, -1, -2))
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)

        # Use dropout with deterministic flag (compatible with remat/checkpoint)
        if self.attn_drop > 0.:
            attn_weights = nn.Dropout(self.attn_drop)(attn_weights, deterministic=deterministic)

        x = jnp.matmul(attn_weights, v)
        x = jnp.transpose(x, (0, 2, 1, 3)).reshape(B, N, C)
        x = TorchLinear(self.dim, bias=True, dtype=self.dtype, param_dtype=self.param_dtype)(x)

        if self.proj_drop > 0.:
            x = nn.Dropout(self.proj_drop)(x, deterministic=deterministic)
        return x, qk

class LightningDiTBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    use_qknorm: bool = False
    use_swiglu: bool = False
    use_rmsnorm: bool = False
    cond_dim: Optional[int] = None
    use_rope: bool = False
    attn_fp32: bool = True
    # Always explicit (no None): either fp32 or bf16.
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x, c, deterministic=True):
        
        if self.use_rmsnorm:
            norm1 = RMSNorm(self.hidden_size)
            norm2 = RMSNorm(self.hidden_size)
        else:
            # PyTorch: elementwise_affine=False
            norm1 = nn.LayerNorm(epsilon=1e-6, use_scale=False, use_bias=False)
            norm2 = nn.LayerNorm(epsilon=1e-6, use_scale=False, use_bias=False)
            
        attn = Attention(
            dim=self.hidden_size,
            num_heads=self.num_heads,
            qkv_bias=True,
            qk_norm=self.use_qknorm,
            use_rmsnorm=self.use_rmsnorm,
            use_rope=self.use_rope,
            attn_fp32=self.attn_fp32,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        
        # [Logic Fix]: Clean MLP selection
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        if self.use_swiglu:
            hid_size = int(2/3 * mlp_hidden_dim)
            hid_size = (hid_size + 31) // 32 * 32
            mlp = SwiGLUFFN(self.hidden_size, hid_size, dtype=self.dtype, param_dtype=self.param_dtype)
        else:
            # Capture from outer scope (self inside StandardMLP refers to StandardMLP, not the block)
            hidden_size_captured = self.hidden_size
            dtype_captured = self.dtype
            param_dtype_captured = self.param_dtype
            class StandardMLP(nn.Module):
                @nn.compact
                def __call__(self, x):
                    h = TorchLinear(mlp_hidden_dim, bias=True, dtype=dtype_captured, param_dtype=param_dtype_captured)(x)
                    h = nn.gelu(h, approximate=False)
                    return TorchLinear(hidden_size_captured, bias=True, dtype=dtype_captured, param_dtype=param_dtype_captured)(h)
            mlp = StandardMLP()

        # AdaLN Modulation - use fp32 for precision on scale/shift/gate values
        out_dim = 6 * self.hidden_size
        adaLN_mod = nn.Sequential([
            nn.silu,
            # Initialize to zeros for stability; use fp32 to avoid bf16 precision loss
            TorchLinear(out_dim, bias=True, weight_init="zeros", bias_init="zeros",
                        dtype=jnp.float32, param_dtype=self.param_dtype)
        ])

        # Compute in fp32, then cast back to model dtype
        chunks = adaLN_mod(c.astype(jnp.float32)).astype(self.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(chunks, 6, axis=1)

        x_norm = norm1(x)
        x_norm = modulate(x_norm, shift_msa, scale_msa)
        if self.dtype is not None:
            x_norm = x_norm.astype(self.dtype)
            c = c.astype(self.dtype)
        x = x + jnp.expand_dims(gate_msa, 1) * attn(x_norm, deterministic=deterministic)[0]
        
        x_norm = norm2(x)
        x_norm = modulate(x_norm, shift_mlp, scale_mlp)
        if self.dtype is not None:
            x_norm = x_norm.astype(self.dtype)
        x = x + jnp.expand_dims(gate_mlp, 1) * mlp(x_norm)
        return x

class FinalLayer(nn.Module):
    hidden_size: int
    patch_size: int
    out_channels: int
    use_rmsnorm: bool = False
    cond_dim: Optional[int] = None
    # Always explicit (no None): either fp32 or bf16.
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, c):
        if self.use_rmsnorm:
            norm_final = RMSNorm(self.hidden_size)
        else:
            norm_final = nn.LayerNorm(epsilon=1e-6, use_scale=False, use_bias=False)

        # AdaLN Modulation - use fp32 for precision on scale/shift values
        adaLN_mod = nn.Sequential([
            nn.silu,
            TorchLinear(2 * self.hidden_size, bias=True, weight_init="zeros", bias_init="zeros",
                        dtype=jnp.float32, param_dtype=self.param_dtype)
        ])

        # Compute in fp32, then cast back to model dtype
        chunks = adaLN_mod(c.astype(jnp.float32)).astype(self.dtype)
        shift, scale = jnp.split(chunks, 2, axis=1)

        x = modulate(norm_final(x), shift, scale)
        x = TorchLinear(self.patch_size * self.patch_size * self.out_channels, bias=True, weight_init="zeros", bias_init="zeros", dtype=self.dtype, param_dtype=self.param_dtype)(x)
        return x

# -----------------------------------------------------------------------------
# 3. Main Model (Fixed Pos Embed & Unpatchify)
# -----------------------------------------------------------------------------
class LightningDiT(nn.Module):
    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 32
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    out_channels: int = 32
    use_qknorm: bool = False
    use_swiglu: bool = False
    use_rope: bool = False
    use_rmsnorm: bool = False
    cond_dim: Optional[int] = None
    n_cls_tokens: int = 0
    attn_fp32: bool = True
    # Always explicit (no None): either fp32 or bf16.
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    # Activation checkpointing (remat) to save memory (at the cost of extra compute)
    use_remat: bool = False
    
    @nn.compact
    def __call__(self, x, c, deterministic=True):
        # Input x is in BHWC format: [B, H, W, C]
        B, H, W, C = x.shape
        p = self.patch_size
        
        # Target grid based on input_size (determines seq_len)
        target_grid = self.input_size // p
        num_patches = target_grid * target_grid
        effective_p = H // target_grid
        grid_h, grid_w = target_grid, target_grid
        
        # Patch Embed (Linear projection of flattened patches)
        # x: [B, H, W, C] = [B, grid_h*effective_p, grid_w*effective_p, C]
        x = x.reshape(B, grid_h, effective_p, grid_w, effective_p, C)
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5)) # [B, grid_h, grid_w, effective_p, effective_p, C]
        x = x.reshape(B, num_patches, effective_p * effective_p * C)
        x = TorchLinear(self.hidden_size, bias=True, dtype=self.dtype, param_dtype=self.param_dtype)(x)

        pos_embed = self.param(
            'pos_embed', 
            sincos_init(self.hidden_size, num_patches), 
            (1, num_patches, self.hidden_size)
        )
        x = (x + pos_embed).astype(self.dtype)
        

        
        # Class Tokens (Concat)
        if self.n_cls_tokens > 0:
            if self.dtype is not None:
                c = c.astype(self.dtype)
            c_tokens = TorchLinear(self.hidden_size, bias=True, dtype=self.dtype, param_dtype=self.param_dtype)(c) 
            c_tokens = jnp.expand_dims(c_tokens, 1) 
            c_tokens = jnp.tile(c_tokens, (1, self.n_cls_tokens, 1)) 
            
            cls_embed = self.param('cls_embed', nn.initializers.normal(stddev=0.02), (1, self.n_cls_tokens, self.hidden_size))
            c_tokens = c_tokens + cls_embed
            x = jnp.concatenate([c_tokens, x], axis=1)
            x = x.astype(self.dtype)
        
        BlockCls = LightningDiTBlock
        if self.use_remat:
            BlockCls = nn.remat(
                LightningDiTBlock,
                prevent_cse=True,
            )

        for i in range(self.depth):
            x = BlockCls(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                use_qknorm=self.use_qknorm,
                use_swiglu=self.use_swiglu,
                use_rmsnorm=self.use_rmsnorm,
                cond_dim=self.cond_dim,
                use_rope=self.use_rope,
                attn_fp32=self.attn_fp32,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f'blocks_{i}'
            )(x, c, deterministic)

        # [B, N, p*p*C] -> [B, H, W, C] (BHWC format)
        # Output uses input_size and patch_size (not effective_p)
        out_size = self.input_size
        x = FinalLayer(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
            use_rmsnorm=self.use_rmsnorm,
            cond_dim=self.cond_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x, c)

        if self.n_cls_tokens > 0:
            x = x[:, self.n_cls_tokens:, :]

        x = x.reshape(B, grid_h, grid_w, p, p, self.out_channels)
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        x = x.reshape(B, out_size, out_size, self.out_channels)
        return x

class TimestepEmbedder(nn.Module):
    hidden_size: int
    frequency_embedding_size: int = 256
    # Always explicit (no None): either fp32 or bf16.
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, t):
        half = self.frequency_embedding_size // 2
        freqs = jnp.exp(
            -math.log(10000) * jnp.arange(0, half, dtype=jnp.float32) / half
        )
        args = t[:, None].astype(jnp.float32) * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if self.frequency_embedding_size % 2:
             embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
             
        t_emb = nn.Sequential([
            TorchLinear(self.hidden_size, bias=True, weight_init="normal", dtype=self.dtype, param_dtype=self.param_dtype),
            nn.silu,
            TorchLinear(self.hidden_size, bias=True, weight_init="normal", dtype=self.dtype, param_dtype=self.param_dtype)
        ])(embedding)
        t_emb = t_emb.astype(self.dtype)
        return t_emb

# -----------------------------------------------------------------------------
# 4. DitGen Wrapper (Fixed Args & Noise Dict)
# -----------------------------------------------------------------------------

class DitGen(nn.Module):
    cond_dim: int
    num_classes: int = 1001
    noise_classes: int = 0
    noise_coords: int = 1
    input_size: int = 32
    in_channels: int = 3
    n_cls_tokens: int = 0
    patch_size: int = 2

    # LightningDiT params
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    out_channels: int = 3
    use_qknorm: bool = False
    use_swiglu: bool = False
    use_rope: bool = False
    use_rmsnorm: bool = False
    use_bf16: bool = False
    attn_fp32: bool = True
    # Activation checkpointing (remat) to save memory (at the cost of extra compute)
    use_remat: bool = False

    def dummy_input(self):
        return {
            'c': jnp.ones(1, dtype=jnp.int32),
            'cfg_scale': 1.0,
            'temp': 1.0,
            'deterministic': True,
        }
    
    def rng_keys(self):
        return ['noise']

    def setup(self):
        dtype = jnp.bfloat16 if self.use_bf16 else jnp.float32
        param_dtype = jnp.float32
        
        self.class_embed = nn.Embed(
            self.num_classes,
            self.cond_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=dtype,
            param_dtype=param_dtype,
            name='Embed_0'
        )

        if self.noise_classes > 0:
            for i in range(self.noise_coords):
                embed = nn.Embed(self.noise_classes, self.cond_dim, 
                             embedding_init=nn.initializers.normal(stddev=0.02),
                             dtype=dtype,
                             param_dtype=param_dtype,
                             name=f'noise_embeds_{i}')
                setattr(self, f'noise_embeds_{i}', embed)
        self.cfg_embedder = TimestepEmbedder(
            self.cond_dim, 
            dtype=dtype, 
            param_dtype=param_dtype,
            name='TimestepEmbedder_0'
        )
        self.cfg_norm = RMSNorm(
            self.cond_dim,
            name='RMSNorm_0'
        )

        self.model = LightningDiT(
            input_size=self.input_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            hidden_size=self.hidden_size,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            out_channels=self.out_channels,
            use_qknorm=self.use_qknorm,
            use_swiglu=self.use_swiglu,
            use_rope=self.use_rope,
            use_rmsnorm=self.use_rmsnorm,
            cond_dim=self.cond_dim,
            n_cls_tokens=self.n_cls_tokens,
            attn_fp32=self.attn_fp32,
            dtype=dtype,
            param_dtype=param_dtype,
            use_remat=self.use_remat,
            name='LightningDiT_0'
        )

    def generate_image(self, x, cond, deterministic=True):
        return self.model(x, cond, deterministic=deterministic)

    def c_cfg_noise_to_cond(self, c, cfg_scale, noise_labels):
        B = c.shape[0]
        cond = self.class_embed(c)
        if self.noise_classes > 0:
            noise_labels = noise_labels + jnp.zeros_like(c, dtype=noise_labels.dtype)[:, None] # move sharding!
            for i in range(self.noise_coords):
                embed = getattr(self, f'noise_embeds_{i}')
                cond = cond + embed(noise_labels[:, i])
        
        if isinstance(cfg_scale, (float, int)):
            cfg_scale_t = jnp.full((B,), cfg_scale)
        else:
            cfg_scale_t = jnp.array(cfg_scale)
            if cfg_scale_t.ndim == 0:
                cfg_scale_t = jnp.tile(jnp.expand_dims(cfg_scale_t, 0), (B,))
            cfg_scale_t = cfg_scale_t + jnp.zeros_like(c, dtype=cfg_scale_t.dtype) 
        cfg_scale_t = self.cfg_norm(self.cfg_embedder(cfg_scale_t))
        cond = cond + cfg_scale_t * 0.02

        if self.use_bf16:
            cond = cond.astype(jnp.bfloat16)
        return cond

    def __call__(self, c, cfg_scale=1.0, temp=1.0, deterministic=True, train=False):
        B = c.shape[0]
        # Noise generation
        rng = self.make_rng('noise')
        rng_x, rng_labels = random.split(rng)
        c = enforce_ddp(c)
        
        if B % jax.device_count() != 0:
            x = random.normal(rng_x, (B, self.input_size, self.input_size, self.in_channels))
        else:
            x = ddp_rand_func(shard="ddp", rand_type="normal")(rng_x, (B, self.input_size, self.input_size, self.in_channels))
        x = x * temp + jnp.zeros_like(c, dtype=x.dtype)[:, None, None, None] # move sharding!

        if self.use_bf16:
            x = x.astype(jnp.bfloat16)
            
        noise_labels = random.randint(rng_labels, (B, max(1, self.noise_coords)), 0, max(1, self.noise_classes)) 
        noise_labels = noise_labels + jnp.zeros_like(c, dtype=noise_labels.dtype)[:, None] # move sharding!
        
        cond = self.c_cfg_noise_to_cond(c, cfg_scale, noise_labels)

        samples = self.generate_image(x, cond, deterministic=deterministic)  # Output is already BHWC
        
        noise_dict = {"x": x}
        noise_dict["noise_labels"] = noise_labels


        return {
            "samples": samples,  # Already in BHWC format
            "noise": noise_dict, 
        }



def build_generator_from_config(model_config: Dict[str, Any]) -> DitGen:
    """Build DitGen directly from a full config dict (e.g., from artifact metadata)."""
    return DitGen(**dict(model_config))


def load_hf(
    name: str,
    *,
    dir: str = HF_ROOT,
):
    """Load generator artifact from HF and return (model, params, metadata).

    The model is reconstructed from ``model_config`` stored in the artifact's
    ``metadata.json``—no preset name or override kwargs needed.
    """
    from models.hf import load_generator_jax

    return load_generator_jax(
        name=name,
        repo_id=HF_REPO_ID,
        prefix=None,
        output_root=dir,
    )
