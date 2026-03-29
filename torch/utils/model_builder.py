"""Factory helpers for building models, datasets, optimizer and logger."""
from __future__ import annotations
import math
from pathlib import Path
import torch
import torch.optim as optim

from dataset.dataset import create_imagenet_split
from utils.logging import WandbLogger
from utils.misc import EasyDict


def create_learning_rate_fn(
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    lr_schedule: str = "const",
):
    """Return a callable ``step -> lr`` implementing the requested schedule.

    Supported schedules: ``"const"`` (constant), ``"cosine"`` / ``"cos"``.
    """

    def lr_fn(step: int) -> float:
        step = int(step)
        if step < warmup_steps:
            return learning_rate * max(step, 1) / max(warmup_steps, 1)
        if lr_schedule in ("cosine", "cos"):
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        return learning_rate

    return lr_fn


def build_model_dict(config, model_class, *, workdir: str = "runs") -> EasyDict:
    """Build model, datasets, optimizer, and logger from a config EasyDict.

    Args:
        config:      EasyDict loaded from a YAML config file.
        model_class: ``torch.nn.Module`` subclass constructor.
        workdir:     Root directory for checkpoints and logs.

    Returns:
        EasyDict with keys: model, optimizer, logger, eval_loader, train_loader,
        dataset_name, preprocess_fn, postprocess_fn, train, learning_rate_fn, feature.
    """
    from utils.distributed import get_world_size

    # ------------------------------------------------------------------ model
    print("Building model...")
    model = model_class(
        num_classes=config.dataset.num_classes,
        **dict(config.model),
    )

    # ---------------------------------------------------------------- dataset
    print("Building dataset...")
    world_size = get_world_size()
    batch_size_per_proc = config.dataset.batch_size // world_size
    eval_batch_per_proc = config.dataset.eval_batch_size // world_size
    resolution = int(config.dataset.resolution)
    use_aug = bool(config.dataset.get("use_aug", False))
    use_latent = bool(config.dataset.get("use_latent", False))
    use_cache = bool(config.dataset.get("use_cache", False))
    dataset_kwargs = dict(config.dataset.get("kwargs", {}))

    train_loader, preprocess_fn, postprocess_fn = create_imagenet_split(
        resolution=resolution,
        use_aug=use_aug,
        use_latent=use_latent,
        use_cache=use_cache,
        batch_size=batch_size_per_proc,
        split="train",
        **dataset_kwargs,
    )
    eval_loader, _, _ = create_imagenet_split(
        resolution=resolution,
        use_aug=use_aug,
        use_latent=use_latent,
        use_cache=use_cache,
        batch_size=eval_batch_per_proc,
        split="val",
        **dataset_kwargs,
    )

    # --------------------------------------------------------------- optimizer
    lr_cfg = config.optimizer.lr_schedule
    learning_rate_fn = create_learning_rate_fn(
        learning_rate=lr_cfg.learning_rate,
        warmup_steps=lr_cfg.warmup_steps,
        total_steps=lr_cfg.total_steps,
        lr_schedule=lr_cfg.get("lr_schedule", "const"),
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr_cfg.learning_rate,
        weight_decay=float(config.optimizer.get("weight_decay", 0.0)),
        betas=(float(config.optimizer.adam_b1), float(config.optimizer.adam_b2)),
    )

    # ----------------------------------------------------------------- logger
    logger = WandbLogger()
    w_cfg = EasyDict(dict(config.get("logging", {})))
    use_wandb = bool(w_cfg.pop("use_wandb", config.get("use_wandb", True)))
    output_root = Path(workdir).resolve()
    logger.set_logging(
        config=config,
        use_wandb=use_wandb,
        workdir=str(output_root),
        **w_cfg,
    )

    return EasyDict(
        model=model,
        optimizer=optimizer,
        logger=logger,
        eval_loader=eval_loader,
        train_loader=train_loader,
        dataset_name=f"imagenet{resolution}",
        preprocess_fn=preprocess_fn,
        postprocess_fn=postprocess_fn,
        train=config.train,
        learning_rate_fn=learning_rate_fn,
        feature=config.get("feature", {}),
    )
