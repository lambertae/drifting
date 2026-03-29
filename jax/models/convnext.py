"""ConvNeXt feature model (JAX)."""

import re
from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange
import torch

from utils.hsdp_util import enforce_ddp, get_global_mesh
from utils.logging import log_for_0


class ConvNextLayerNorm(nn.Module):
    """LayerNorm on the last channel for NHWC tensors."""

    normalized_shape: int
    eps: float = 1e-6

    def setup(self):
        self.weight = self.param("weight", nn.initializers.ones, (self.normalized_shape,))
        self.bias = self.param("bias", nn.initializers.zeros, (self.normalized_shape,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        old_dtype = x.dtype
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        x = self.weight * x + self.bias
        return x.astype(old_dtype)


class ConvNextGRN(nn.Module):
    """Global Response Normalization."""

    dim: int
    eps: float = 1e-6

    def setup(self):
        self.gamma = self.param("gamma", nn.initializers.zeros, (1, 1, 1, self.dim))
        self.beta = self.param("beta", nn.initializers.zeros, (1, 1, 1, self.dim))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        old_dtype = x.dtype
        norm = jnp.sum(x ** 2, axis=(1, 2), keepdims=True)
        gx = jnp.sqrt(norm + self.eps)
        nx = gx / (jnp.mean(gx, axis=-1, keepdims=True) + self.eps)
        return (self.gamma * (x * nx) + self.beta + x).astype(old_dtype)


class ConvNextBlock(nn.Module):
    """ConvNeXtV2 residual block."""

    dim: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dwconv = nn.Conv(
            features=self.dim,
            kernel_size=(7, 7),
            padding="SAME",
            feature_group_count=self.dim,
            name="dwconv",
            dtype=self.dtype,
        )
        self.norm = ConvNextLayerNorm(self.dim, eps=1e-6)
        self.pwconv1 = nn.Dense(features=4 * self.dim, name="pwconv1", dtype=self.dtype)
        self.grn = ConvNextGRN(4 * self.dim)
        self.pwconv2 = nn.Dense(features=self.dim, name="pwconv2", dtype=self.dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = jax.nn.gelu(x, approximate=False)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


def safe_std(x, axis, eps=1e-6):
    """Stable standard deviation in fp32."""
    x32 = x.astype(jnp.float32)
    mean = jnp.mean(x32, axis=axis, keepdims=True)
    var = jnp.mean((x32 - mean) ** 2, axis=axis, keepdims=False)
    var = jnp.maximum(var, 0.0)
    return jnp.sqrt(var + eps)


class ConvNextV2(nn.Module):
    """ConvNeXtV2 backbone with activation export."""

    in_chans: int = 3
    num_classes: int = 1000
    drop_path_rate: float = 0.0
    head_init_scale: float = 1.0
    depths: Sequence[int] = (3, 3, 9, 3)
    dims: Sequence[int] = (96, 192, 384, 768)
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        stem = nn.Sequential(
            layers=[
                nn.Conv(features=self.dims[0], kernel_size=(4, 4), strides=(4, 4), dtype=self.dtype),
                ConvNextLayerNorm(self.dims[0], eps=1e-6),
            ],
            name="downsample_layers_0",
        )
        layers = [stem]
        for i in range(3):
            downsample_layer = nn.Sequential(
                layers=[
                    ConvNextLayerNorm(self.dims[i], eps=1e-6),
                    nn.Conv(features=self.dims[i + 1], kernel_size=(2, 2), strides=(2, 2), dtype=self.dtype),
                ],
                name=f"downsample_layers_{i + 1}",
            )
            layers.append(downsample_layer)
        self.downsample_layers = layers

        stages = []
        for i in range(4):
            stage = nn.Sequential(
                layers=[ConvNextBlock(dim=self.dims[i], dtype=self.dtype) for _ in range(self.depths[i])],
                name=f"stages_{i}",
            )
            stages.append(stage)
        self.stages = stages
        self.norm = nn.LayerNorm(epsilon=1e-6)
        self.head = nn.Dense(features=self.num_classes, dtype=self.dtype)

    def get_activations(self, x: jnp.ndarray) -> dict:
        """Return multi-scale normalized feature maps for drift loss."""
        x = enforce_ddp(x)
        x = jax.image.resize(x, shape=(x.shape[0], 224, 224, 3), method="bilinear").astype(self.dtype)
        feature_dict = {}

        def normalize(y):
            old_dtype = y.dtype
            y = y.astype(jnp.float32)
            y = (y - y.mean(axis=-1, keepdims=True)) / (y.std(axis=-1, keepdims=True) + 1e-3)
            return y.astype(old_dtype)

        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            x_normed = normalize(x)
            if i > 0:
                feature_dict[f"convenxt_stage_{i}"] = rearrange(x_normed, "b h w c -> b (h w) c")
            feature_dict[f"convenxt_stage_{i}_mean"] = x_normed.mean(axis=(1, 2))[:, None, :]
            feature_dict[f"convenxt_stage_{i}_std"] = safe_std(rearrange(x_normed, "b h w c -> b (h w) c"), axis=1)[:, None, :]
        feature_dict["global_mean"] = self.norm(x.mean(axis=(1, 2)))[:, None, :]
        feature_dict["global_std"] = safe_std(rearrange(normalize(x), "b h w c -> b (h w) c"), axis=1)[:, None, :]
        return feature_dict

    def forward_features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward backbone and return pooled representation."""
        x = jax.image.resize(x, shape=(x.shape[0], 224, 224, 3), method="bilinear").astype(self.dtype)
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean(axis=(1, 2)))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Default forward returns pooled feature vector."""
        return self.forward_features(x)


ConvNextBase = partial(ConvNextV2, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
ConvNextTiny = partial(ConvNextV2, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])


def convert_weights_to_jax(jax_params: dict, module_pt, hf: bool = False):
    """Convert PyTorch ConvNeXt weights to JAX parameter layout."""
    log_for_0("Converting ConvNext weights to jax...")
    jax_params_flat, jax_param_pytree = jax.tree_util.tree_flatten_with_path(jax_params)
    pt_params = {path: param for path, param in module_pt.items()}

    if hf:
        new_pt_params = {}
        for path, param in pt_params.items():
            path = re.sub(r"classifier\.", "head.", path)
            path = re.sub(r"convnextv2\.encoder\.", "", path)
            path = re.sub(r"convnextv2\.embeddings\.patch_embeddings\.", "downsample_layers_0.layers_0.", path)
            path = re.sub(r"convnextv2\.embeddings\.layernorm\.", "downsample_layers_0.layers_1.", path)
            path = re.sub(r"stages\.([0-3])\.downsampling_layer\.(\d+)", lambda m: f"downsample_layers_{m.group(1)}.layers_{m.group(2)}", path)
            path = re.sub(r"stages\.([0-3])\.layers\.(\d+)", lambda m: f"stages_{m.group(1)}.layers_{m.group(2)}", path)
            path = re.sub(r"layernorm", "norm", path)
            path = re.sub(r"convnextv2\.", "", path)
            path = re.sub(r"grn\.weight", "grn.gamma", path)
            path = re.sub(r"grn\.bias", "grn.beta", path)
            new_pt_params[path] = param
        pt_params = new_pt_params
    else:
        new_pt_params = {}
        for path, param in pt_params.items():
            for i in range(4):
                path = re.sub(rf"stages\.{i}\.(\d+)", lambda m: f"stages_{i}.layers_{m.group(1)}", path)
                path = re.sub(rf"downsample_layers\.{i}\.(\d+)", lambda m: f"downsample_layers_{i}.layers_{m.group(1)}", path)
            new_pt_params[path] = param
        pt_params = new_pt_params

    pt_params = {f"params.{path}": param for path, param in pt_params.items()}
    direct_copy = ["grn"]
    pt_params_flat = []

    for path, param in jax_params_flat:
        shape = param.shape
        path = ".".join([p.key for p in path])
        path = re.sub(r"\.scale|.kernel", ".weight", path)
        if path in pt_params:
            pt_param = pt_params[path]
            if any(dc_key in path for dc_key in direct_copy):
                pt_params_flat.append(jnp.asarray(pt_param.detach().numpy()))
            else:
                if len(shape) == 4:
                    pt_param = torch.permute(pt_param, (2, 3, 1, 0))
                else:
                    pt_param = torch.permute(pt_param, tuple(reversed(range(len(shape)))))
                pt_params_flat.append(jnp.asarray(pt_param.detach().numpy()))
            pt_params.pop(path)
        else:
            log_for_0(f"[WARNING] missing param '{path}' with shape {shape} from PyTorch model")
            pt_params_flat.append(None)

    for path, param in pt_params.items():
        log_for_0(f"[WARNING] params not loaded '{path}' with shape {param.shape} from PyTorch model")

    log_for_0("ConvNext conversion done.")
    return jax.tree_util.tree_unflatten(jax_param_pytree, pt_params_flat)


def load_convnext_jax_model(model_name: str = "base", use_bf16: bool = False):
    """Load ConvNeXt from HF PyTorch weights and return sharded JAX params."""
    if model_name == "base":
        model_jax = ConvNextBase(dtype=jnp.bfloat16 if use_bf16 else jnp.float32)
        model_load_name = "facebook/convnextv2-base-22k-224"
    elif model_name == "tiny":
        model_jax = ConvNextTiny(dtype=jnp.bfloat16 if use_bf16 else jnp.float32)
        model_load_name = "facebook/convnextv2-tiny-22k-224"
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    dummy_input = jnp.ones((1, 224, 224, 3))
    jax_params = model_jax.init(jax.random.PRNGKey(0), dummy_input)

    from transformers import ConvNextV2ForImageClassification

    model_pt = ConvNextV2ForImageClassification.from_pretrained(model_load_name).state_dict()
    jax_params = convert_weights_to_jax(jax_params, model_pt, hf=True)

    import os
    import shutil
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache/huggingface")
    if os.path.exists(cache_dir):
        log_for_0("Removing Huggingface cache directory...")
        shutil.rmtree(cache_dir)

    from jax.sharding import NamedSharding, PartitionSpec as P

    mesh = get_global_mesh()
    jax_params = jax.tree.map(lambda x: jax.device_put(x, NamedSharding(mesh, P())), jax_params)
    return model_jax, jax_params
