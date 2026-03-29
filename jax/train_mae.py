from __future__ import annotations

import argparse
import copy
import gc
import os
import time
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm
import jax.experimental.multihost_utils as mu

from dataset.dataset import epoch0_sampler, infinite_sampler
from models.mae_model import MAEResNetJAX
from utils.ckpt_util import restore_checkpoint, save_checkpoint, save_params_ema_artifact
from utils.env import HF_ROOT
from utils.hsdp_util import (
    data_shard,
    ddp_shard,
    init_state_from_dummy_input,
    map_to_sharding,
    merge_data,
    pad_and_merge,
    set_global_mesh,
)
from utils.init_util import maybe_init_state_params
from utils.logging import is_rank_zero, log_for_0
from utils.misc import load_config, prepare_rng, profile_func, run_init
from utils.model_builder import build_model_dict


run_init()


class TrainState(train_state.TrainState):
    ema_params: Optional[Any] = None
    ema_decay: float = 0.999


def input_dict(batch):
    """Convert preprocessed batch dict to model forward kwargs."""
    return {"x": batch["images"], "labels": batch["labels"]}


def train_step(
    state: TrainState,
    batch,
    *,
    rng_init: jax.Array,
    forward_dict: dict,
    step_keys=("dropout", "masking"),
    learning_rate_fn: Any,
    preprocess_fn: Any,
    max_grad_norm: float = 2.0,
):
    """Run one MAE optimization step.

    Args:
        state: MAE TrainState.
        batch: dict with `images` shape `(B, H, W, C)` and `labels` shape `(B,)`.
        rng_init: base PRNGKey for this train loop.
        forward_dict: kwargs forwarded into the MAE model call.
        step_keys: RNG stream names used inside the MAE model.
        learning_rate_fn: schedule mapping step -> scalar lr.
        preprocess_fn: batch preprocessing callable returning the same batch structure.
        max_grad_norm: gradient clipping norm.
    """
    batch = jax.tree_util.tree_map(lambda x: jax.lax.with_sharding_constraint(x, ddp_shard()), batch)
    batch = preprocess_fn(batch)
    batch = jax.tree_util.tree_map(lambda x: jax.lax.with_sharding_constraint(x, data_shard()), batch)

    rng_step = jax.random.fold_in(rng_init, state.step)
    forward_kwargs = input_dict(batch)

    def loss_fn(params):
        loss, metric = state.apply_fn(
            {"params": params},
            train=True,
            rngs=prepare_rng(rng_step, step_keys),
            **forward_kwargs,
            **forward_dict,
        )
        return loss.mean(), metric

    (loss, metric), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    g_norm = optax.global_norm(grads)
    updates, _ = optax.clip_by_global_norm(max_grad_norm).update(grads, None)
    new_state = state.apply_gradients(grads=updates)

    new_ema_params = jax.tree.map(
        lambda ema, p: ema * state.ema_decay + p * (1.0 - state.ema_decay),
        state.ema_params,
        new_state.params,
    )
    new_state = new_state.replace(ema_params=new_ema_params)

    metric["loss"] = loss
    metric["lr"] = learning_rate_fn(state.step)
    metric["g_norm"] = g_norm
    metric = jax.tree.map(lambda x: x.mean(), metric)
    return new_state, metric


def eval_step(
    params,
    batch,
    rng_step,
    *,
    apply_fn,
    forward_dict,
    step_keys=("dropout", "masking"),
    preprocess_fn: Any,
):
    """Single MAE eval step."""
    batch = jax.tree_util.tree_map(lambda x: jax.lax.with_sharding_constraint(x, ddp_shard()), batch)
    batch = preprocess_fn(batch)
    batch = jax.tree_util.tree_map(lambda x: jax.lax.with_sharding_constraint(x, data_shard()), batch)
    loss, metric = apply_fn(
        {"params": params},
        **input_dict(batch),
        **forward_dict,
        train=False,
        rngs=prepare_rng(rng_step, step_keys),
    )
    metric["loss"] = loss
    return metric


def eval_loop(
    state: TrainState,
    eval_loader,
    eval_step_func,
    *,
    eval_samples=5000,
    forward_dict=None,
    use_ema=False,
    rng_eval=None,
    ema_to_params_func=lambda x: x,
):
    """Evaluate MAE for model/EMA params."""
    forward_dict = forward_dict or {}
    rng_eval = jax.random.PRNGKey(0) if rng_eval is None else rng_eval
    mu.sync_global_devices("eval loop started")
    eval_iter = epoch0_sampler(eval_loader)
    params = ema_to_params_func(state.ema_params) if use_ema else state.params

    all_metrics = []
    masks = []
    goal_bsz = 0
    n_samples = 0
    for i, batch in enumerate(eval_iter):
        # Lock eval to the first seen batch size so later short batches can be padded consistently.
        if i == 0:
            goal_bsz = jax.tree.leaves(batch)[0].shape[0]
        batch, mask = pad_and_merge(batch, goal_bsz, use_ddp=True)
        rng_eval_step = jax.random.fold_in(rng_eval, i)
        metric = eval_step_func(params, batch, rng_eval_step, forward_dict=dict(forward_dict))

        # Trim the final step logically via the mask instead of changing the padded batch shape.
        if n_samples + mask.shape[0] > eval_samples:
            keep_n = eval_samples - n_samples
            mask = jnp.concatenate([mask[:keep_n], jnp.zeros(mask.shape[0] - keep_n)], axis=0)
        n_samples += mask.shape[0]

        all_metrics.append(metric)
        masks.append(mask)
        if n_samples >= eval_samples:
            break

    masks = jnp.concatenate(masks, axis=0)
    all_metrics = jax.tree.map(lambda *x: (jnp.concatenate(x, axis=0) * masks).mean() / (masks.mean() + 1e-8), *all_metrics)
    mu.sync_global_devices("eval loop finished")
    return all_metrics


def train_mae(
    *,
    model,  # MAEResNetJAX instance to train
    optimizer,  # Optax optimizer transform
    logger,  # logger with log_dict / finish
    eval_loader,  # evaluation dataloader iterator source
    train_loader,  # training dataloader iterator source
    learning_rate_fn,  # callable(step) -> lr
    forward_dict,  # kwargs passed to MAE forward in train
    eval_forward_dict,  # kwargs passed to MAE forward in eval
    preprocess_fn,  # preprocessing function for dataloader batches
    postprocess_fn,  # kept for interface compatibility (unused)
    total_steps=100000,  # max optimization steps
    save_per_step=10000,  # checkpoint save interval
    eval_per_step=2000,  # evaluation interval
    eval_samples=5000,  # number of eval samples
    ema_decay=0.999,  # single EMA decay
    seed=42,  # global RNG seed
    finetune_last_steps=0,  # enable cls finetune in last N steps
    warmup_finetune=1000,  # cls-finetune warmup steps
    finetune_cls=0.5,  # target lambda_cls at finetune end
    max_grad_norm=2.0,  # gradient clipping norm
    keep_every=500000,  # long-term checkpoint retention interval
    keep_last=2,  # number of latest checkpoints to keep
    init_from="",  # optional `hf://<name>` or local artifact dir
    workdir="runs",  # run root containing checkpoints/logs
    model_config=None,  # model config dict saved with EMA metadata
):
    """MAE training loop (ported from nnflow_jax, simplified infra)."""
    del postprocess_fn

    if isinstance(ema_decay, (list, tuple)):
        if len(ema_decay) != 1:
            raise ValueError(f"Expected a single ema_decay value, got {ema_decay}")
        ema_decay = float(ema_decay[0])
    else:
        ema_decay = float(ema_decay)

    rng = jax.random.PRNGKey(seed)
    rng, _ = jax.random.split(rng)
    rng_train, rng_eval = jax.random.split(rng)

    state = init_state_from_dummy_input(
        model,
        optimizer,
        TrainState,
        rng,
        model.dummy_input(),
        ["dropout", "masking"],
        ema_decay=ema_decay,
    )
    ema_to_params_func = map_to_sharding(state.params)

    state = restore_checkpoint(state=state, workdir=workdir)
    if int(jax.device_get(state.step)) == 0 and init_from:
        log_for_0("Initializing MAE params from init_from=%s", init_from)
        state = maybe_init_state_params(
            state,
            model_type="mae",
            init_from=init_from,
            hf_cache_dir=HF_ROOT,
        )
    eval_step_jit = jax.jit(
        lambda params, batch, rng_step, forward_dict: eval_step(
            params,
            batch,
            rng_step,
            apply_fn=state.apply_fn,
            preprocess_fn=preprocess_fn,
            forward_dict=forward_dict,
        )
    )
    train_step_jit = jax.jit(
        lambda state_, batch_, forward_dict_: train_step(
            state_,
            batch_,
            rng_init=rng_train,
            learning_rate_fn=learning_rate_fn,
            preprocess_fn=preprocess_fn,
            max_grad_norm=max_grad_norm,
            forward_dict=forward_dict_,
        ),
        out_shardings=(jax.tree.map(lambda x: x.sharding, state), None),
    )

    forward_zeros_dict = copy.deepcopy(forward_dict)
    forward_zeros_dict["mask_ratio_min"] = 0.0
    forward_zeros_dict["mask_ratio_max"] = 0.0

    log_for_0("Starting MAE training loop...")
    step = int(state.step)
    initial_step = step
    pbar = tqdm(range(step, total_steps), initial=step, total=total_steps) if is_rank_zero() else range(step, total_steps)
    train_iter = infinite_sampler(train_loader, step)
    mu.sync_global_devices("train loop started")

    start_finetune_step = total_steps - finetune_last_steps
    start_time_all = time.time()
    for step in pbar:
        start_time = time.time()
        logger.set_step(step)

        batch = next(train_iter)
        batch = merge_data(batch, use_ddp=True)
        finish_prepare = time.time()

        cur_dict = dict(copy.deepcopy(forward_dict))
        if step >= start_finetune_step:
            cur_dict["lambda_cls"] = finetune_cls * min(1.0, (step - start_finetune_step) / max(1, warmup_finetune))

        profile_metrics = {}
        if step == initial_step:
            profile_metrics = profile_func(train_step_jit, (state, batch, cur_dict), name="train_step")

        new_state, metrics = train_step_jit(state, batch, cur_dict)
        metrics = jax.tree.map(lambda x: x.mean(), metrics)

        finish_train = time.time()
        metrics["kimg"] = (step - initial_step + 1) * jax.tree.leaves(batch)[0].shape[0] / 1000.0
        metrics["time/total"] = finish_train - start_time
        metrics["time/prepare"] = finish_prepare - start_time
        metrics["time/train"] = finish_train - finish_prepare
        metrics["time/per_step"] = (finish_train - start_time_all) / (step - initial_step + 1)
        metrics.update(profile_metrics)
        logger.log_dict(metrics)

        state = new_state
        step += 1

        if step % eval_per_step == 0:
            eval_metrics = eval_loop(
                state,
                eval_loader,
                eval_step_jit,
                eval_samples=eval_samples,
                forward_dict=eval_forward_dict,
                rng_eval=rng_eval,
                use_ema=False,
                ema_to_params_func=ema_to_params_func,
            )
            logger.log_dict_dir("eval", eval_metrics)
            eval_metrics_ema = eval_loop(
                state,
                eval_loader,
                eval_step_jit,
                eval_samples=eval_samples,
                forward_dict=eval_forward_dict,
                rng_eval=rng_eval,
                use_ema=True,
                ema_to_params_func=ema_to_params_func,
            )
            logger.log_dict_dir(f"eval_ema_{state.ema_decay:g}", eval_metrics_ema)

            eval_metrics_nomask = eval_loop(
                state,
                eval_loader,
                eval_step_jit,
                eval_samples=eval_samples,
                forward_dict=forward_zeros_dict,
                rng_eval=rng_eval,
                use_ema=False,
                ema_to_params_func=ema_to_params_func,
            )
            logger.log_dict_dir("eval_nomask", eval_metrics_nomask)
            eval_metrics_nomask_ema = eval_loop(
                state,
                eval_loader,
                eval_step_jit,
                eval_samples=eval_samples,
                forward_dict=forward_zeros_dict,
                rng_eval=rng_eval,
                use_ema=True,
                ema_to_params_func=ema_to_params_func,
            )
            logger.log_dict_dir(f"eval_ema_{state.ema_decay:g}_nomask", eval_metrics_nomask_ema)

        if (step in [total_steps, start_finetune_step]) or (step % save_per_step == 0 and step < start_finetune_step):
            save_checkpoint(state, keep=keep_last, keep_every=keep_every, workdir=workdir)
            save_params_ema_artifact(
                state,
                workdir=workdir,
                kind="mae",
                model_config=model_config,
            )

        if step % 100 == 0:
            mu.sync_global_devices(f"train step {step} finished")

    mu.sync_global_devices("train loop finished")
    logger.finish()
    del model, optimizer, eval_loader, train_loader, state
    gc.collect()
    jax.clear_caches()
    mu.sync_global_devices("train loop cleanup finished")


def main_mae(config, output_dir="runs"):
    """Build MAE model pipeline and launch MAE training."""
    set_global_mesh(config.get("hsdp_dim", jax.local_device_count()))
    if "logging" not in config:
        config.logging = {}
    config.logging.name = Path(output_dir).resolve().name

    model_dict = build_model_dict(config, MAEResNetJAX, workdir=output_dir)
    train_mae(
        model=model_dict.model,
        optimizer=model_dict.optimizer,
        logger=model_dict.logger,
        eval_loader=model_dict.eval_loader,
        train_loader=model_dict.train_loader,
        learning_rate_fn=model_dict.learning_rate_fn,
        preprocess_fn=model_dict.preprocess_fn,
        postprocess_fn=model_dict.postprocess_fn,
        model_config=dict(config.model),
        workdir=output_dir,
        **config.train,
    )


def main(args):
    """CLI entrypoint for MAE training."""
    config = load_config(args.config)
    main_mae(config, output_dir=args.workdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to MAE config.")
    parser.add_argument("--workdir", type=str, default="runs", help="Local workdir root for checkpoints/logs.")
    args = parser.parse_args()
    args.output_dir = args.workdir
    main(args)
