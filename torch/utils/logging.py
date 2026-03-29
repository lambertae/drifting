"""Logging utilities – WandbLogger and NullLogger."""
from __future__ import annotations
from typing import Any, Dict, Optional
import torch


class NullLogger:
    """No-op logger used when W&B is disabled or on non-main ranks."""

    def log_dict(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        pass

    def log(self, key: str, value: Any, step: Optional[int] = None) -> None:
        pass

    def finish(self) -> None:
        pass

    def set_logging(self, **kwargs) -> None:
        pass


class WandbLogger:
    """Thin wrapper around wandb with graceful fallback to NullLogger."""

    def __init__(self):
        self._run = None
        self._enabled = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_logging(
        self,
        config=None,
        use_wandb: bool = True,
        workdir: str = "runs",
        project: str = "drifting",
        name: Optional[str] = None,
        entity: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialise wandb if `use_wandb` is True and we are on the main rank."""
        from utils.distributed import is_main_process

        if not use_wandb or not is_main_process():
            return

        try:
            import wandb

            init_kwargs: Dict[str, Any] = dict(
                project=project,
                dir=workdir,
                config=dict(config) if config is not None else None,
            )
            if name is not None:
                init_kwargs["name"] = name
            if entity is not None:
                init_kwargs["entity"] = entity
            init_kwargs.update(kwargs)

            self._run = wandb.init(**init_kwargs)
            self._enabled = True
        except Exception as exc:
            print(f"[WandbLogger] wandb.init failed – logging disabled. Reason: {exc}", flush=True)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_python(value: Any) -> Any:
        """Convert tensors / numpy arrays to plain Python scalars."""
        if isinstance(value, torch.Tensor):
            return value.detach().float().mean().item()
        try:
            import numpy as np
            if isinstance(value, np.ndarray):
                return float(value.mean())
            if isinstance(value, (np.floating, np.integer)):
                return value.item()
        except ImportError:
            pass
        return value

    def log_dict(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        from utils.distributed import is_main_process

        if not is_main_process():
            return

        clean = {k: self._to_python(v) for k, v in metrics.items()}
        if self._enabled and self._run is not None:
            self._run.log(clean, step=step)
        else:
            parts = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in clean.items())
            print(f"[metrics] step={step} | {parts}", flush=True)

    def log(self, key: str, value: Any, step: Optional[int] = None) -> None:
        self.log_dict({key: value}, step=step)

    def finish(self) -> None:
        if self._enabled and self._run is not None:
            self._run.finish()
            self._run = None
            self._enabled = False
