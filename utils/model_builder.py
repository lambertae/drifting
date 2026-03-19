from pathlib import Path

import jax
import optax

from dataset.dataset import create_imagenet_split
from utils.logging import WandbLogger
from utils.misc import EasyDict


def create_learning_rate_fn(
    learning_rate,
    warmup_steps,
    total_steps,
    lr_schedule="const",
):
    """Create warmup + main learning-rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=1e-6,
        end_value=learning_rate,
        transition_steps=warmup_steps,
    )
    if lr_schedule in ["cosine", "cos"]:
        cosine_steps = max(total_steps - warmup_steps, 1)
        schedule_fn = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=cosine_steps,
            alpha=1e-6,
        )
    elif lr_schedule == "const":
        schedule_fn = optax.constant_schedule(value=learning_rate)
    else:
        raise NotImplementedError(lr_schedule)

    return optax.join_schedules(
        schedules=[warmup_fn, schedule_fn],
        boundaries=[warmup_steps],
    )


def build_model_dict(config, model_class, *, workdir: str = "runs"):
    """Build model, datasets, optimizer, and logger from config."""
    print("Building model...")
    model = model_class(
        num_classes=config.dataset.num_classes,
        **config.model,
    )

    print("Building dataset...")
    batch_size_per_node = config.dataset.batch_size // jax.process_count()
    resolution = int(config.dataset.resolution)
    use_aug = bool(config.dataset.get("use_aug", False))
    use_latent = bool(config.dataset.get("use_latent", False))
    use_cache = bool(config.dataset.get("use_cache", False))

    train_loader, preprocess_fn, postprocess_fn = create_imagenet_split(
        resolution=resolution,
        use_aug=use_aug,
        use_latent=use_latent,
        use_cache=use_cache,
        batch_size=batch_size_per_node,
        split="train",
        **config.dataset.kwargs,
    )

    eval_loader, _, _ = create_imagenet_split(
        resolution=resolution,
        use_aug=use_aug,
        use_latent=use_latent,
        use_cache=use_cache,
        batch_size=config.dataset.eval_batch_size // jax.process_count(),
        split="val",
        **config.dataset.kwargs,
    )

    learning_rate_fn = create_learning_rate_fn(**config.optimizer.lr_schedule)

    optimizer = optax.adamw(
        learning_rate=learning_rate_fn,
        weight_decay=config.optimizer.get("weight_decay", 0.0),
        b1=config.optimizer.adam_b1,
        b2=config.optimizer.adam_b2,
    )

    logger = WandbLogger()
    w_cfg = EasyDict(dict(config.get("logging", {})))
    use_wandb = bool(w_cfg.get("use_wandb", config.get("use_wandb", True)))
    if "use_wandb" in w_cfg:
        del w_cfg["use_wandb"]
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
