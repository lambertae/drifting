"""Minimal Hugging Face helpers for Drift artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from flax import serialization
from utils.env import HF_ROOT


def read_metadata(artifact_dir: Path) -> Dict[str, Any]:
    # Used by utils/init_util.py for local artifact restore, and by the HF loaders below.
    """Read metadata.json from artifact directory."""
    return json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))


def load_jax_ema_params(artifact_dir: Path) -> Any:
    # Used by utils/init_util.py for local artifact restore, and by the HF loaders below.
    """Load ema params msgpack from artifact directory."""
    return serialization.msgpack_restore((artifact_dir / "ema_params.msgpack").read_bytes())


def _download_artifact(
    *,
    repo_id: str,
    kind: str,
    backend: str,
    model_id: str,
    output_root: str,
    prefix: Optional[str],
) -> Path:
    # Internal helper used by load_mae_jax/load_generator_jax to materialize HF artifacts locally.
    """Download artifact folder from HF and return resolved local directory."""
    from huggingface_hub import snapshot_download

    local_root = Path(output_root).resolve() / "models" / kind / backend / model_id
    local_root.mkdir(parents=True, exist_ok=True)
    root = f"models/{kind}/{backend}/{model_id}"
    path_in_repo = f"{prefix.strip('/')}/{root}" if prefix else root

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{path_in_repo}/*"],
        local_dir=str(local_root),
    )
    nested = local_root / path_in_repo
    return nested if nested.exists() else local_root


def load_mae_jax(
    name: str,
    *,
    repo_id: str,
    prefix: Optional[str] = None,
    output_root: str = HF_ROOT,
) -> Tuple[Any, Any, Dict[str, Any]]:
    # Used by models.mae_model.load_mae_hf -> utils.init_util load path for HF MAE restore.
    """Load MAE model config+params from HF artifact."""
    artifact_dir = _download_artifact(
        repo_id=repo_id,
        kind="mae",
        backend="jax",
        model_id=name,
        output_root=output_root,
        prefix=prefix,
    )
    metadata = read_metadata(artifact_dir)

    from models.mae_model import _mae_from_metadata

    module = _mae_from_metadata(metadata)
    params = load_jax_ema_params(artifact_dir)
    return module, params, metadata


def load_generator_jax(
    name: str,
    *,
    repo_id: str,
    prefix: Optional[str] = None,
    output_root: str = HF_ROOT,
) -> Tuple[Any, Any, Dict[str, Any]]:
    # Used by models.generator.load_hf -> utils.init_util load path for HF generator restore/infer.
    """Load generator model config+params from HF artifact.

    The model is reconstructed from ``model_config`` in the artifact's
    ``metadata.json``—no preset name needed.
    """
    artifact_dir = _download_artifact(
        repo_id=repo_id,
        kind="gen",
        backend="jax",
        model_id=name,
        output_root=output_root,
        prefix=prefix,
    )
    metadata = read_metadata(artifact_dir)

    model_cfg = dict(metadata.get("model_config", {}) or {})
    if not model_cfg:
        raise ValueError(
            f"Generator artifact is missing metadata.model_config and cannot be restored: {name}"
        )
    from models.generator import build_generator_from_config

    module = build_generator_from_config(model_cfg)
    params = load_jax_ema_params(artifact_dir)
    return module, params, metadata
