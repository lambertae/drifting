from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import flax
import jax
import jax.numpy as jnp
from flax.training import checkpoints

from models.hf import load_jax_ema_params, read_metadata
from utils.env import HF_ROOT


def resolve_artifact_dir(path: str) -> Path:
    base = Path(path).resolve()
    params_ema_dir = base / "params_ema"
    ckpt_dir = base / "checkpoints"
    if params_ema_dir.is_dir():
        return params_ema_dir
    if ckpt_dir.is_dir():
        return ckpt_dir
    return base


def put_like(tree: Any, template: Any) -> Any:
    """Place each leaf of ``tree`` on the same device/sharding as ``template``."""
    def _put(x, t):
        if isinstance(t, jax.Array):
            return jax.device_put(jnp.asarray(x), t.sharding)
        return x

    return jax.tree.map(_put, tree, template)


def _load_local_init_entry(path: str) -> Tuple[Any, Dict[str, Any]]:
    artifact_dir = resolve_artifact_dir(path)
    metadata_path = artifact_dir / "metadata.json"
    params_path = artifact_dir / "ema_params.msgpack"
    legacy_meta_path = artifact_dir / "ema_model.metadata.json"
    legacy_params_path = artifact_dir / "ema_model.msgpack"

    if metadata_path.is_file() and params_path.is_file():
        return load_jax_ema_params(artifact_dir), read_metadata(artifact_dir)
    if params_path.is_file():
        return load_jax_ema_params(artifact_dir), {}
    if legacy_meta_path.is_file() and legacy_params_path.is_file():
        metadata = json.loads(legacy_meta_path.read_text(encoding="utf-8"))
        params = flax.serialization.msgpack_restore(legacy_params_path.read_bytes())
        return params, metadata
    if legacy_params_path.is_file():
        params = flax.serialization.msgpack_restore(legacy_params_path.read_bytes())
        return params, {}

    restored = checkpoints.restore_checkpoint(str(artifact_dir), target=None, step=None)
    if isinstance(restored, dict) and "params" in restored:
        return restored["params"], {}

    raise ValueError(
        "Local init_from must be an artifact or checkpoint dir with params: "
        f"{artifact_dir}"
    )


def load_init_entry(
    model_type: str,
    init_from: str,
    *,
    hf_cache_dir: str = HF_ROOT,
) -> Tuple[Any, Dict[str, Any]]:
    """Load params+metadata for `mae` or `generator` from HF or local path."""
    if not init_from:
        raise ValueError("`init_from` is empty.")

    if not init_from.startswith("hf://"):
        return _load_local_init_entry(init_from)

    model_name = init_from[len("hf://") :].strip()
    if not model_name:
        raise ValueError("Invalid HF init_from path, expected `hf://<name>`.")

    if model_type == "mae":
        from models.mae_model import load_mae_hf

        _, params, metadata = load_mae_hf(
            model_name,
            dir=hf_cache_dir,
        )
        params = params["params"] if isinstance(params, dict) and "params" in params else params
        return params, metadata

    if model_type == "generator":
        from models.generator import load_hf

        _, params, metadata = load_hf(
            model_name,
            dir=hf_cache_dir,
        )
        params = params["params"] if isinstance(params, dict) and "params" in params else params
        return params, metadata

    raise ValueError(f"Unsupported model_type={model_type!r}, expected 'mae' or 'generator'.")


def maybe_init_state_params(
    state: Any,
    *,
    model_type: str,
    init_from: str,
    hf_cache_dir: str = HF_ROOT,
) -> Any:
    """Initialize `state.params` and EMA params from external source when requested."""
    if not init_from:
        return state

    loaded_params, _ = load_init_entry(
        model_type,
        init_from,
        hf_cache_dir=hf_cache_dir,
    )
    params = put_like(loaded_params, state.params)
    ema_params = params
    return state.replace(params=params, ema_params=ema_params)


def load_generator_model_and_params(
    init_from: str,
    *,
    hf_cache_dir: str = HF_ROOT,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load a generator model+params pair from ``hf://...`` or a local artifact path.

    The model is reconstructed from ``model_config`` stored in the artifact's
    ``metadata.json``—no preset name or override kwargs needed.
    """
    if not init_from:
        raise ValueError("`init_from` is empty.")

    if init_from.startswith("hf://"):
        from models.generator import load_hf

        model_name = init_from[len("hf://") :].strip()
        model, params, metadata = load_hf(
            model_name,
            dir=hf_cache_dir,
        )
        params = params["params"] if isinstance(params, dict) and "params" in params else params
        return model, params, metadata

    params, metadata = _load_local_init_entry(init_from)
    model_cfg = dict(metadata.get("model_config", {}) or {})
    if not model_cfg:
        raise ValueError(
            f"missing metadata.model_config: local artifact at {Path(init_from).resolve()} "
            "cannot be restored without model_config in metadata.json"
        )
    from models.generator import build_generator_from_config

    model = build_generator_from_config(model_cfg)
    params = jax.tree.map(jnp.asarray, params)
    return model, params, metadata
