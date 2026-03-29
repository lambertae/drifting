"""Checkpoint utilities for PyTorch Drift training."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional
import torch
from utils.distributed import log_for_0


def _output_root(workdir: Optional[str] = None) -> Path:
    if workdir:
        return Path(workdir).resolve()
    return Path("runs").resolve()


def _job_ckpt_dir(workdir: Optional[str] = None) -> Path:
    return _output_root(workdir) / "checkpoints"


def save_checkpoint(
    state_dict: Dict[str, Any],
    step: int,
    workdir: Optional[str] = None,
    keep: int = 2,
    keep_every: Optional[int] = None,
) -> Path:
    """Serialize `state_dict` to disk and prune old checkpoints.

    Args:
        state_dict: arbitrary dict passed to ``torch.save``.
        step:       training step used to name the file.
        workdir:    root directory for runs; defaults to ``./runs``.
        keep:       number of most-recent checkpoints to retain.
        keep_every: additionally keep checkpoints whose step is divisible by this value.

    Returns:
        Path to the saved file.
    """
    ckpt_dir = _job_ckpt_dir(workdir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    path = ckpt_dir / f"ckpt_{step:08d}.pt"
    torch.save(state_dict, path)
    log_for_0("Saved checkpoint step %d to %s", step, str(path))

    # Prune old checkpoints
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
    if keep_every:
        keep_set = {c for c in ckpts if int(c.stem.split("_")[1]) % keep_every == 0}
    else:
        keep_set: set = set()

    to_keep = keep_set | set(ckpts[-keep:])
    for c in ckpts:
        if c not in to_keep:
            c.unlink(missing_ok=True)

    return path


def restore_checkpoint(
    workdir: Optional[str] = None,
    step: Optional[int] = None,
    map_location: str = "cpu",
) -> Optional[Dict[str, Any]]:
    """Load a checkpoint from disk.

    Args:
        workdir:      root directory; defaults to ``./runs``.
        step:         specific step to restore; if None restores the latest.
        map_location: passed to ``torch.load``.

    Returns:
        The loaded state dict, or None if no checkpoint is found.
    """
    ckpt_dir = _job_ckpt_dir(workdir)
    if not ckpt_dir.exists():
        log_for_0("No checkpoint dir at %s", str(ckpt_dir))
        return None

    if step is not None:
        path = ckpt_dir / f"ckpt_{step:08d}.pt"
        if not path.exists():
            log_for_0("Checkpoint not found: %s", str(path))
            return None
    else:
        ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
        if not ckpts:
            return None
        path = ckpts[-1]

    log_for_0("Restoring checkpoint from %s", str(path))
    return torch.load(path, map_location=map_location)


def save_ema_artifact(
    ema_state_dict: Dict[str, Any],
    step: int,
    workdir: Optional[str] = None,
    kind: str = "gen",
    model_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save an EMA parameter snapshot alongside JSON metadata.

    Args:
        ema_state_dict: model state dict (EMA weights).
        step:           training step.
        workdir:        root directory.
        kind:           artifact tag (e.g. "gen", "mae").
        model_config:   arbitrary config dict stored in metadata.json.

    Returns:
        Path to the artifact directory.
    """
    out_dir = _output_root(workdir) / "params_ema"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(ema_state_dict, out_dir / "ema_params.pt")

    metadata = {
        "format": "torch",
        "kind": kind,
        "backend": "torch",
        "step": step,
        "path": "params_ema/ema_params.pt",
        "model_config": dict(model_config or {}),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    log_for_0("Saved EMA artifact step %d to %s", step, str(out_dir))
    return out_dir
