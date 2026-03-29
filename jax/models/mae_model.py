"""Self-contained JAX MAE-ResNet (no model indirection)."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn
from utils.env import HF_REPO_ID, HF_ROOT
from utils.init_util import load_init_entry, put_like


def _choose_gn_groups(num_channels: int, max_groups: int = 32) -> int:
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return max(g, 1)


class _BasicBlock(nn.Module):
    """ResNet basic block: two 3x3 convs with GroupNorm, ReLU, and residual skip.

    Input/output shape: ``(B, H, W, C)``; spatial dims may be halved when ``stride=2``.
    """
    filters: int
    in_channels: Optional[int] = None
    stride: int = 1
    gn_max_groups: int = 32
    dropout_prob: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv1 = nn.Conv(self.filters, kernel_size=(3, 3), strides=self.stride, padding=1, use_bias=False, dtype=self.dtype)
        self.gn1 = nn.GroupNorm(num_groups=_choose_gn_groups(self.filters, self.gn_max_groups), dtype=self.dtype)
        self.conv2 = nn.Conv(self.filters, kernel_size=(3, 3), strides=1, padding=1, use_bias=False, dtype=self.dtype)
        self.gn2 = nn.GroupNorm(num_groups=_choose_gn_groups(self.filters, self.gn_max_groups), dtype=self.dtype)
        self.drop = nn.Dropout(self.dropout_prob)
        self.proj_conv = nn.Conv(self.filters, kernel_size=(1, 1), strides=self.stride, use_bias=False, dtype=self.dtype)
        self.proj_gn = nn.GroupNorm(num_groups=_choose_gn_groups(self.filters, self.gn_max_groups), dtype=self.dtype)
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        residual = x
        y = self.conv1(x)
        y = self.gn1(y)
        y = nn.relu(y)
        y = self.drop(y, deterministic=not train)
        y = self.conv2(y)
        y = self.gn2(y)

        if residual.shape != y.shape:
            residual = self.proj_conv(residual)
            residual = self.proj_gn(residual)

        return nn.relu(residual + y)


class _ResNetEncoder(nn.Module):
    """4-stage ResNet encoder producing multi-scale feature maps.

    Returns a dict ``{conv1, layer1, …, layer4}`` each with shape ``(B, H_i, W_i, C_i)``.
    """
    base_channels: int = 64
    layers: Tuple[int, int, int, int] = (2, 2, 2, 2)
    dropout_prob: float = 0.0
    gn_max_groups: int = 32
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv1 = nn.Conv(self.base_channels, kernel_size=(3, 3), strides=1, padding=1, use_bias=False, dtype=self.dtype)
        self.gn1 = nn.GroupNorm(num_groups=_choose_gn_groups(self.base_channels, self.gn_max_groups), dtype=self.dtype)

        stages = []
        ch = self.base_channels
        for stage_idx, num_blocks in enumerate(self.layers):
            stride = 2 if stage_idx > 0 else 1
            out_ch = ch * (2 ** stage_idx) if stage_idx > 0 else ch
            blocks = []
            blocks.append(
                _BasicBlock(
                    out_ch,
                    in_channels=ch if stage_idx == 0 else (ch * (2 ** (stage_idx - 1))),
                    stride=stride,
                    dropout_prob=self.dropout_prob,
                    dtype=self.dtype,
                )
            )
            for _ in range(1, num_blocks):
                blocks.append(
                    _BasicBlock(
                        out_ch,
                        in_channels=out_ch,
                        stride=1,
                        dropout_prob=self.dropout_prob,
                        dtype=self.dtype,
                    )
                )
            stages.append(nn.Sequential(blocks))
            setattr(
                self,
                f"layer{stage_idx + 1}_norm",
                nn.GroupNorm(num_groups=_choose_gn_groups(out_ch, self.gn_max_groups), dtype=self.dtype),
            )
        self.stages = stages

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        train: bool,
        return_block_outputs: bool = False,
    ):
        feats: Dict[str, jnp.ndarray] = {}
        block_outputs: Dict[str, List[jnp.ndarray]] = {}
        x = self.conv1(x)
        x = self.gn1(x)
        x = nn.relu(x)
        feats["conv1"] = x

        for i, blocks in enumerate(self.stages):
            layer_name = f"layer{i + 1}"
            outs: List[jnp.ndarray] = []
            for block in blocks.layers:
                x = block(x, train=train)
                outs.append(x)
            block_outputs[layer_name] = outs
            norm_layer = getattr(self, f"{layer_name}_norm")
            x = norm_layer(x)
            feats[layer_name] = x
        if return_block_outputs:
            return feats, block_outputs
        return feats


class _ConvGNReLU(nn.Module):
    channels: int
    kernel: int = 3
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(self.channels, kernel_size=(self.kernel, self.kernel), padding=self.kernel // 2, use_bias=False, dtype=self.dtype)
        self.gn = nn.GroupNorm(num_groups=_choose_gn_groups(self.channels, 32), dtype=self.dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.relu(self.gn(self.conv(x)))


class _UpBlock(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.concat_norm_fn = nn.GroupNorm(num_groups=32, dtype=self.dtype)
        self.proj = _ConvGNReLU(self.out_channels, kernel=3, dtype=self.dtype)
        self.refine = _ConvGNReLU(self.out_channels, kernel=3, dtype=self.dtype)

    def __call__(self, x: jnp.ndarray, skip: jnp.ndarray) -> jnp.ndarray:
        b, h, w, c = x.shape
        x = jax.image.resize(x, shape=(b, skip.shape[1], skip.shape[2], c), method="bilinear")
        x = jnp.concatenate([x, skip], axis=-1)
        x = self.concat_norm_fn(x)
        x = self.proj(x)
        x = self.refine(x)
        return x


class _UNetDecoder(nn.Module):
    base_channels: int
    out_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c1 = self.base_channels
        c2 = self.base_channels
        c3 = self.base_channels * 2
        c4 = self.base_channels * 4
        c5 = self.base_channels * 8
        self.bridge = _ConvGNReLU(c5, dtype=self.dtype)
        self.up43 = _UpBlock(c4, dtype=self.dtype)
        self.up32 = _UpBlock(c3, dtype=self.dtype)
        self.up21 = _UpBlock(c2, dtype=self.dtype)
        self.up10 = _UpBlock(c1, dtype=self.dtype)
        self.head = nn.Conv(self.out_channels, kernel_size=(1, 1), dtype=self.dtype)

    def __call__(self, feats: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        x = self.bridge(feats["layer4"])
        x = self.up43(x, feats["layer3"])
        x = self.up32(x, feats["layer2"])
        x = self.up21(x, feats["layer1"])
        x = self.up10(x, feats["conv1"])
        return self.head(x)


def patch_input(x: jnp.ndarray, input_patch_size: int) -> jnp.ndarray:
    return rearrange(
        x,
        "b (h1 h2) (w1 w2) c -> b h1 w1 (h2 w2 c)",
        h2=input_patch_size,
        w2=input_patch_size,
    )


def make_patch_mask(x: jnp.ndarray, rng: jax.Array, mask_ratio: jnp.ndarray, patch_size: int = 4) -> jnp.ndarray:
    b, h, w, _ = x.shape
    nh, nw = h // patch_size, w // patch_size
    noise = jax.random.uniform(rng, (b, nh, nw), dtype=x.dtype)
    mask = (noise < mask_ratio[:, None, None]).astype(x.dtype)
    mask = jnp.repeat(mask, patch_size, axis=1)
    mask = jnp.repeat(mask, patch_size, axis=2)
    return mask[..., None]


def safe_std(x: jnp.ndarray, axis, eps: float = 1e-6, keepdims: bool = False) -> jnp.ndarray:
    x32 = x.astype(jnp.float32)
    mean = jnp.mean(x32, axis=axis, keepdims=True)
    var = jnp.mean((x32 - mean) ** 2, axis=axis, keepdims=keepdims)
    return jnp.sqrt(jnp.maximum(var, 0.0) + eps)


class MAEResNetJAX(nn.Module):
    num_classes: int = 1000
    in_channels: int = 3
    base_channels: int = 64
    patch_size: int = 4
    dropout_prob: float = 0.0
    layers: Tuple[int, int, int, int] = (2, 2, 2, 2)
    use_bf16: bool = False
    input_patch_size: int = 1

    def setup(self):
        self.dtype = jnp.bfloat16 if self.use_bf16 else jnp.float32
        self.encoder = _ResNetEncoder(
            base_channels=self.base_channels,
            layers=self.layers,
            dropout_prob=self.dropout_prob,
            dtype=self.dtype,
        )
        self.decoder = _UNetDecoder(
            base_channels=self.base_channels,
            out_channels=self.in_channels * self.input_patch_size * self.input_patch_size,
            dtype=self.dtype,
        )
        self.fc = nn.Dense(self.num_classes, dtype=self.dtype)

    def __call__(
        self,
        x: jnp.ndarray,
        labels: jnp.ndarray,
        *,
        lambda_cls: float = 0.0,
        mask_ratio_min: float = 0.75,
        mask_ratio_max: float = 0.75,
        train: bool = True,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        x = x.astype(self.dtype)
        x = patch_input(x, self.input_patch_size)
        ratio_rng, mask_rng = jax.random.split(self.make_rng("masking"))
        b = x.shape[0]
        mask_ratio = jax.random.uniform(ratio_rng, (b,), dtype=self.dtype) * (mask_ratio_max - mask_ratio_min) + mask_ratio_min
        mask = make_patch_mask(x, mask_rng, mask_ratio, self.patch_size)
        x_in = x * (1.0 - mask)

        feats = self.encoder(x_in, train=train)
        top = feats["layer4"]
        pooled = jnp.mean(top, axis=(1, 2))
        logits = self.fc(pooled)
        recon = self.decoder(feats)

        one_hot = jax.nn.one_hot(labels, self.num_classes, dtype=self.dtype)
        cls_loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1)
        mse = (recon - x) ** 2
        recon_loss = (mse * mask).sum(axis=(1, 2, 3)) / (mask.sum(axis=(1, 2, 3)) + 1e-8)
        loss = lambda_cls * cls_loss + (1.0 - lambda_cls) * recon_loss
        metrics = {
            "loss": loss,
            "cls_loss": cls_loss,
            "recon_loss": recon_loss,
            "accuracy": (jnp.argmax(logits, axis=-1) == labels).astype(self.dtype),
            "mask_ratio": mask.mean(axis=(1, 2, 3)),
        }
        return loss, metrics

    def get_activations(
        self,
        x: jnp.ndarray,
        *,
        patch_mean_size: Optional[List[int]] = [2,4],
        patch_std_size: Optional[List[int]] = [2,4],
        use_std: bool = True,
        use_mean: bool = True,
        every_k_block: float = 2,
    ) -> Dict[str, jnp.ndarray]:
        """Extract multi-scale features from the encoder for drift loss.

        Args:
            x: input images of shape ``(B, H, W, C)``.

        Returns:
            Dict of named feature tensors.  Each value has shape
            ``(B, T, D)`` where ``T`` is the spatial token count and ``D``
            the channel dimension.  Keys include ``conv1``, ``layer{1-4}``,
            and optional aggregated variants (``*_mean``, ``*_std``,
            ``*_mean_{size}``, ``*_std_{size}``).
        """
        patch_mean_size = patch_mean_size or []
        patch_std_size = patch_std_size or []

        x = x.astype(self.dtype)
        x = patch_input(x, self.input_patch_size)
        need_blocks = isinstance(every_k_block, (int, float)) and not math.isinf(float(every_k_block)) and every_k_block >= 1
        if need_blocks:
            feats, block_outputs = self.encoder(x, train=False, return_block_outputs=True)
        else:
            feats = self.encoder(x, train=False)
            block_outputs = {}

        out: Dict[str, jnp.ndarray] = {}
        out["norm_x"] = jnp.sqrt((x ** 2).mean(axis=(1, 2)) + 1e-6)[:, None, :]

        def process_feat(name: str, feat: jnp.ndarray) -> None:
            b, h, w, c = feat.shape
            out[name] = rearrange(feat, "b h w c -> b (h w) c")
            if use_mean:
                out[f"{name}_mean"] = feat.mean(axis=(1, 2))[:, None, :]
            if use_std:
                out[f"{name}_std"] = safe_std(feat, axis=(1, 2))[:, None, :]

            for size in patch_mean_size:
                if h % size == 0 and w % size == 0:
                    reshaped = rearrange(feat, "b (h s1) (w s2) c -> b (h w) (s1 s2) c", s1=size, s2=size)
                    out[f"{name}_mean_{size}"] = reshaped.mean(axis=2)

            for size in patch_std_size:
                if h % size == 0 and w % size == 0:
                    reshaped = rearrange(feat, "b (h s1) (w s2) c -> b (h w) (s1 s2) c", s1=size, s2=size)
                    out[f"{name}_std_{size}"] = safe_std(reshaped, axis=2)

        for name, feat in feats.items():
            process_feat(name, feat)

        if need_blocks:
            k = int(every_k_block)
            for i in range(1, 5):
                lname = f"layer{i}"
                blocks = block_outputs.get(lname, [])
                for blk_idx, feat_i in enumerate(blocks, start=1):
                    if blk_idx % k == 0:
                        process_feat(f"{lname}_blk{blk_idx}", feat_i)

        return out

    def dummy_input(self) -> Dict[str, Any]:
        p = self.input_patch_size
        return {
            "x": jnp.zeros((1, 32 * p, 32 * p, self.in_channels), dtype=jnp.float32),
            "labels": jnp.zeros((1,), dtype=jnp.int32),
            "lambda_cls": 0.0,
            "mask_ratio_min": 0.75,
            "mask_ratio_max": 0.75,
            "train": False,
        }

def load_mae_hf(
    name: str,
    *,
    dir: str = HF_ROOT,
) -> Tuple[MAEResNetJAX, Any, Dict[str, Any]]:
    """Minimal HF loader returning (model, params, metadata).

    Interface is intentionally minimal:
    - name: HF model id under `models/mae/jax/<name>`
    - dir: local download root
    """
    from models.hf import load_mae_jax

    repo_id = HF_REPO_ID
    prefix = None
    model, params, metadata = load_mae_jax(
        name,
        repo_id=repo_id,
        prefix=prefix,
        output_root=dir,
    )
    return model, params, metadata


def _mae_from_metadata(metadata: Dict[str, Any]) -> MAEResNetJAX:
    model_config = dict(metadata.get("model_config", {}) or {})
    num_classes = int(model_config.pop("num_classes", 1000))
    return MAEResNetJAX(num_classes=num_classes, **model_config)


def build_feature_model_and_params(
    path: str = "",
    use_convnext: bool = False,
    convnext_bf16: bool = False,
):
    """Build feature model and params from a local/HF MAE artifact or ConvNeXt."""
    if use_convnext:
        from models.convnext import load_convnext_jax_model

        return load_convnext_jax_model(model_name="base", use_bf16=convnext_bf16)

    if not path:
        raise ValueError("`path` is required when use_convnext=False.")

    from utils.hsdp_util import init_model_distributed

    with jax.default_device(jax.devices("cpu")[0]):
        entry, metadata = load_init_entry(
            "mae",
            path,
            hf_cache_dir=HF_ROOT,
        )
        if not metadata:
            raise ValueError(f"MAE artifact is missing metadata required to rebuild the model: {path}")
        feature_model = _mae_from_metadata(metadata)

    init_params = init_model_distributed(
        feature_model,
        feature_model.dummy_input(),
        rng_keys_extra=["masking", "dropout"],
    )

    merged_params = put_like(entry, init_params["params"])
    return feature_model, merged_params


def build_activation_function(
    mae_path: str = "",
    use_convnext=False,
    convnext_bf16=False,
    use_mae=True,
    postprocess_fn=lambda x: x,
):
    variables = dict()
    if use_mae:
        feature_model, feature_params = build_feature_model_and_params(
            path=mae_path,
        )
        variables["mae_params"] = feature_params

    if use_convnext:
        convnext_model, convnext_feature_params = build_feature_model_and_params(
            use_convnext=True,
            convnext_bf16=convnext_bf16,
        )
        variables["convnext_params"] = convnext_feature_params

    def activation_fn(params, x, convnext_kwargs=dict(), has_scale=False, **kwargs):
        usual_feats = dict()
        usual_feats["global"] = x.reshape(x.shape[0], 1, -1)
        if has_scale:
            usual_feats["norm_x"] = jnp.sqrt((x ** 2).mean(axis=(1, 2)) + 1e-6)[:, None, :]

        if use_mae:
            mae_feats = feature_model.apply({"params": params["mae_params"]}, x, method=feature_model.get_activations, **kwargs)
            usual_feats = {**usual_feats, **mae_feats}

        if use_convnext:
            x = postprocess_fn(x)
            x = x.transpose(0, 2, 3, 1)
            x = (x - jnp.array([0.485, 0.456, 0.406])) / jnp.array([0.229, 0.224, 0.225])
            convnext_feats = convnext_model.apply(params["convnext_params"], x, method=convnext_model.get_activations, **convnext_kwargs)
            usual_feats = {**usual_feats, **convnext_feats}
        return usual_feats

    return activation_fn, variables
