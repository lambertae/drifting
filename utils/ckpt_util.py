from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import jax
import json
from flax import serialization
from flax.training import checkpoints
from jax.experimental import multihost_utils as mu

from utils.logging import log_for_0


def _to_python_int(x) -> int:
    x = jax.device_get(x)
    try:
        return int(x)
    except TypeError:
        return int(x.reshape(-1)[0])


def _output_root(workdir: Optional[str] = None) -> Path:
    if workdir:
        return Path(workdir).resolve()
    return Path("runs").resolve()


def _job_ckpt_dir(workdir: Optional[str] = None) -> Path:
    return _output_root(workdir) / "checkpoints"


def restore_checkpoint(step=None, state=None, workdir: Optional[str] = None):
    ckpt_dir = _job_ckpt_dir(workdir=workdir)
    if not ckpt_dir.exists():
        log_for_0("No local checkpoint dir at %s", str(ckpt_dir))
        return state

    if step is not None:
        step = int(step)

    if state is None:
        return checkpoints.restore_checkpoint(str(ckpt_dir), target=None, step=step)

    target_dict = serialization.to_state_dict(state)
    restored_cpu_dict = checkpoints.restore_checkpoint(str(ckpt_dir), target=None, step=step)
    if restored_cpu_dict is None:
        return state

    # Reshard restored leaves to match target sharding/layout.
    def put_to_device(cpu_leaf, target_leaf):
        if target_leaf is None or cpu_leaf is None:
            return target_leaf
        if hasattr(target_leaf, "sharding"):
            return jax.device_put(cpu_leaf, target_leaf.sharding)
        return cpu_leaf

    sharded_dict = jax.tree.map(
        put_to_device,
        restored_cpu_dict,
        target_dict,
        is_leaf=lambda x: isinstance(x, jax.Array) or (x is None) or (isinstance(x, dict) and len(x) == 0),
    )
    return serialization.from_state_dict(state, sharded_dict)


def save_checkpoint(state, keep=2, keep_every=None, workdir: Optional[str] = None):
    mu.sync_global_devices("save_checkpoint_before_allgather")
    cpu_state = mu.process_allgather(state)
    step = _to_python_int(cpu_state.step)
    cpu_state = cpu_state.replace(step=step)

    ckpt_dir = _job_ckpt_dir(workdir=workdir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_for_0("Saving checkpoint step %d to %s", step, str(ckpt_dir))
    checkpoints.save_checkpoint_multiprocess(
        str(ckpt_dir),
        cpu_state,
        step,
        keep=keep,
        keep_every_n_steps=keep_every,
    )
    mu.sync_global_devices("save_checkpoint_barrier")


def save_params_ema_artifact(
    state: Any,
    *,
    workdir: Optional[str] = None,
    kind: str,
    model_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save the release EMA tree as a standalone restorable artifact.

    This is separate from `checkpoints/` on purpose:
    - `checkpoints/` stores resumable TrainState snapshots.
    - `params_ema/` stores the exported EMA params + metadata used by restore/infer/HF flows.
    """
    cpu_ema = mu.process_allgather(state.ema_params)
    step = _to_python_int(state.step)
    ema_decay = float(getattr(state, "ema_decay"))

    out_dir = _output_root(workdir) / "params_ema"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ema_params.msgpack").write_bytes(serialization.msgpack_serialize(cpu_ema))

    metadata = {
        "format": "flax.msgpack",
        "kind": kind,
        "backend": "jax",
        "ema_decay": ema_decay,
        "step": step,
        "path": "params_ema/ema_params.msgpack",
        "model_config": dict(model_config or {}),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    log_for_0("Saved EMA params artifact step %d to %s", step, str(out_dir))
    return out_dir
