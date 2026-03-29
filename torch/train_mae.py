"""PyTorch MAE training loop for Drift."""
from __future__ import annotations
import copy
import time
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.ckpt_util import restore_checkpoint, save_checkpoint, save_ema_artifact
from utils.distributed import (
    barrier, get_rank, get_world_size, is_main_process, log_for_0, setup_distributed,
)
from utils.misc import EasyDict, load_config


# ---------------------------------------------------------------------------
# EMA helpers (duplicated here to keep modules self-contained)
# ---------------------------------------------------------------------------

def _copy_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone().float() for k, v in model.state_dict().items()}


def _update_ema(ema_params: Dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if k in ema_params:
                ema_params[k].mul_(decay).add_(v.detach().float(), alpha=1.0 - decay)


# ---------------------------------------------------------------------------
# Single MAE training step
# ---------------------------------------------------------------------------

def train_step_mae(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    ema_params: Dict[str, torch.Tensor],
    cfg: EasyDict,
    step: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    """Execute one MAE optimisation step.

    Args:
        model:      MAE model (possibly DDP-wrapped).
        optimizer:  AdamW optimiser.
        images:     Input images [B, C, H, W].
        labels:     Class labels [B].
        ema_params: EMA weight dict (updated in-place).
        cfg:        Training config EasyDict.
        step:       Current global step.
        scaler:     Optional AMP grad scaler.

    Returns:
        Dict of scalar metrics.
    """
    model.train()

    mask_ratio_min = float(cfg.get("mask_ratio_min", 0.5))
    mask_ratio_max = float(cfg.get("mask_ratio_max", 0.75))
    lambda_cls = float(cfg.get("lambda_cls", 0.0))

    with torch.cuda.amp.autocast(enabled=(scaler is not None)):
        loss, metrics_dict = model(
            x=images,
            labels=labels,
            train=True,
            mask_ratio_min=mask_ratio_min,
            mask_ratio_max=mask_ratio_max,
            lambda_cls=lambda_cls,
        )

    optimizer.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
        optimizer.step()

    # EMA update
    raw_model = model.module if isinstance(model, DDP) else model
    _update_ema(ema_params, raw_model, decay=float(cfg.get("ema_decay", 0.9999)))

    out = {"loss": loss.item()}
    out.update({k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in metrics_dict.items()})
    return out


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_mae(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    logger,
    eval_loader,
    train_loader,
    dataset_name: str,
    preprocess_fn: Callable,
    postprocess_fn: Callable,
    train: EasyDict,
    learning_rate_fn: Callable,
    feature: Any = None,
    workdir: str = "runs",
    **_ignored,
) -> None:
    """Full MAE training loop.

    Handles checkpoint restore/save, LR scheduling, periodic logging.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_size = get_world_size()

    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[get_rank()])

    raw_model = model.module if isinstance(model, DDP) else model
    ema_params = _copy_params(raw_model)

    use_amp = torch.cuda.is_available() and bool(train.get("use_amp", True))
    scaler: Optional[torch.cuda.amp.GradScaler] = (
        torch.cuda.amp.GradScaler() if use_amp else None
    )

    total_steps = int(train.get("total_steps", 200_000))
    start_step = 0

    ckpt = restore_checkpoint(workdir=workdir)
    if ckpt is not None:
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        ema_params = ckpt.get("ema_params", ema_params)
        start_step = ckpt.get("step", 0)
        log_for_0("Resumed from step %d", start_step)

    from dataset.dataset import infinite_sampler
    data_iter = infinite_sampler(train_loader, start_step=start_step)

    log_interval = int(train.get("log_interval", 100))
    save_interval = int(train.get("save_interval", 10_000))
    ema_save_interval = int(train.get("ema_save_interval", 50_000))

    t0 = time.time()

    for step in range(start_step, total_steps):
        batch = next(data_iter)
        batch = preprocess_fn(batch)
        images = batch["images"]
        labels = batch["labels"]

        # Learning rate schedule
        lr = learning_rate_fn(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        metrics = train_step_mae(
            model=model,
            optimizer=optimizer,
            images=images,
            labels=labels,
            ema_params=ema_params,
            cfg=train,
            step=step,
            scaler=scaler,
        )
        metrics["lr"] = lr

        if step % log_interval == 0 and is_main_process():
            elapsed = time.time() - t0
            log_for_0(
                "step=%d  loss=%.4f  lr=%.2e  elapsed=%.1fs",
                step, metrics["loss"], lr, elapsed,
            )
            logger.log_dict({f"mae_train/{k}": v for k, v in metrics.items()}, step=step)

        if step > 0 and step % save_interval == 0 and is_main_process():
            state = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema_params": ema_params,
                "step": step,
            }
            save_checkpoint(state, step=step, workdir=workdir, keep=3)

        if step > 0 and step % ema_save_interval == 0 and is_main_process():
            save_ema_artifact(ema_params, step=step, workdir=workdir, kind="mae")

    # Final save
    if is_main_process():
        state = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema_params": ema_params,
            "step": total_steps,
        }
        save_checkpoint(state, step=total_steps, workdir=workdir, keep=3)
        save_ema_artifact(ema_params, step=total_steps, workdir=workdir, kind="mae")

    logger.finish()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def main_mae(config: EasyDict, output_dir: str) -> None:
    from models.mae_model import MAEResNetPyTorch
    from utils.model_builder import build_model_dict

    model_dict = build_model_dict(config, MAEResNetPyTorch, workdir=output_dir)
    train_mae(**model_dict, workdir=output_dir)


def main(args) -> None:
    setup_distributed()
    config = load_config(args.config)
    main_mae(config, output_dir=getattr(args, "output_dir", "runs"))
