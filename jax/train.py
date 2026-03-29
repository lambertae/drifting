import argparse
import gc
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import jax.experimental.multihost_utils as mu
import optax
from flax.training import train_state
from tqdm import tqdm
from einops import repeat, rearrange

from dataset.dataset import infinite_sampler, get_postprocess_fn
from drift_loss import drift_loss
from memory_bank import ArrayMemoryBank
from models.mae_model import build_activation_function
from utils.ckpt_util import save_checkpoint, restore_checkpoint, save_params_ema_artifact
from utils.env import HF_ROOT
from utils.fid_util import evaluate_fid
from utils.hsdp_util import (
    map_to_sharding, data_shard, merge_data, pad_and_merge,
    init_state_from_dummy_input, ddp_shard, set_global_mesh, enforce_ddp,
)
from utils.init_util import maybe_init_state_params
from utils.logging import log_for_0, is_rank_zero
from utils.misc import load_config, prepare_rng, profile_func, run_init
from utils.model_builder import build_model_dict
run_init()

class TrainState(train_state.TrainState):
    ema_params: Optional[Any] = None
    ema_decay: float = 0.999


def _generator_model_config(model) -> dict:
    return {
        name: value
        for name, value in vars(model).items()
        if name not in {"parent", "name"} and not name.startswith("_")
    }


def train_step(state: TrainState, labels, samples, negative_samples, feature_params, feature_apply, rng_init: jax.random.PRNGKey, learning_rate_fn: Any = None, cfg_min=1.0, cfg_max=4.0, neg_cfg_pw=1.0, no_cfg_frac=0.0, gen_per_label=8, activation_kwargs=dict(), loss_kwargs=dict(R_list=[0.02, 0.05, 0.2]), max_grad_norm=2.0):
    """Run one generator optimization step.

    Args:
        state: generator TrainState.
        labels: class labels with shape `(B,)`.
        samples: positive memory-bank samples with shape `(B, P, H, W, C)`.
        negative_samples: negative memory-bank samples with shape `(B, N, H, W, C)`.
        feature_params: feature-model variable tree consumed by `feature_apply`.
        feature_apply: callable returning activation dict for input batch of shape `(B', H, W, C)`.
        rng_init: base PRNGKey for this train loop.
        learning_rate_fn: schedule mapping step -> scalar lr.
        cfg_min: lower bound for sampled CFG scale.
        cfg_max: upper bound for sampled CFG scale.
        neg_cfg_pw: power-law exponent for negative CFG sampling weights.
        no_cfg_frac: probability of replacing sampled CFG with `1.0`.
        gen_per_label: number of generator samples drawn per label, output shape `(B * gen_per_label, H, W, C)`.
        activation_kwargs: keyword args forwarded to feature activation extraction.
        loss_kwargs: keyword args forwarded to `drift_loss`.
        max_grad_norm: gradient clipping norm.
    """
    rng_step = jax.random.fold_in(rng_init, state.step)


    # first: compute cfg
    cfg_seed, rng_step = jax.random.split(rng_step) # [B]
    cfg_seed1, cfg_seed2 = jax.random.split(cfg_seed)
    frac = jax.random.uniform(cfg_seed1, (samples.shape[0],))
    pw = 1 - neg_cfg_pw
    if abs(pw) < 1e-6:
        cfg = jnp.exp(jnp.log(cfg_min) + frac * (jnp.log(cfg_max) - jnp.log(cfg_min)))
    else:
        cfg = (cfg_min ** pw + frac * (cfg_max ** pw - cfg_min ** pw)) ** (1/pw)
    
    frac2 = jax.random.uniform(cfg_seed2, (samples.shape[0],))
    cfg = jnp.where(frac2 < no_cfg_frac, 1.0, cfg)

    def loss_grad_info(labels, samples, negative_samples, cfg, rng_step):
        labels = enforce_ddp(labels)
        samples = enforce_ddp(samples)
        negative_samples = enforce_ddp(negative_samples)
        cfg = enforce_ddp(cfg)
        bsz = labels.shape[0]
        
        uncond_w = (cfg - 1) * (gen_per_label - 1) / max(1, negative_samples.shape[1]) # [B]
        n_pos, n_gen, n_uncond = samples.shape[1], gen_per_label, negative_samples.shape[1]
        neg_samples_input = rearrange(jnp.concatenate([samples, negative_samples], axis=1), 'b x ... -> (b x) ...')
        neg_samples_input = enforce_ddp(neg_samples_input)
        sg_features = jax.lax.stop_gradient(feature_apply(feature_params, neg_samples_input, **activation_kwargs))
        if bsz % jax.device_count() == 0:
            sg_features = jax.tree.map(lambda u: rearrange(u, '(b x) ... -> b x ...', x=n_pos + n_uncond), sg_features) 
        else:
            sg_features = jax.tree.map(lambda u: rearrange(enforce_ddp(u), '(b x) ... -> b x ...', x=n_pos + n_uncond), sg_features) 
        sg_features = enforce_ddp(sg_features)

        def loss_fn(params):
            input_labels = enforce_ddp(repeat(labels, 'b -> (b g)', g=gen_per_label))
            input_cfg = enforce_ddp(repeat(cfg, 'b -> (b g)', g=gen_per_label))
            gen_samples = state.apply_fn(
                {'params': params},
                train=True,
                rngs=prepare_rng(rng_step, ['noise']),
                c=input_labels,
                cfg_scale=input_cfg,
            )['samples'] 
            gen_features = feature_apply(feature_params, gen_samples, **activation_kwargs)
            if bsz % jax.device_count() == 0:
                gen_features = jax.tree.map(lambda u: rearrange(u, '(b g) ... -> b g ...', g=n_gen), gen_features) # [B, G, F, D]
            else:
                gen_features = jax.tree.map(lambda u: rearrange(enforce_ddp(u), '(b g) ... -> b g ...', g=n_gen), gen_features) # [B, G, F, D]
            gen_features = enforce_ddp(gen_features)

            def feature_loss(sg_features, gen_features):
                feature_pos, feature_gen, feature_uncond = sg_features[:, :n_pos], gen_features, sg_features[:, n_pos:]
                feature_pos = enforce_ddp(rearrange(feature_pos, 'b x f d -> (b f) x d'))
                feature_gen = enforce_ddp(rearrange(feature_gen, 'b x f d -> (b f) x d'))
                feature_uncond = enforce_ddp(rearrange(feature_uncond, 'b x f d -> (b f) x d'))
                B = feature_gen.shape[0]
                loss, info = drift_loss(
                    gen=feature_gen,
                    fixed_pos=feature_pos,
                    fixed_neg=feature_uncond,
                    weight_gen=jnp.ones_like(feature_gen[:, :, 0]),
                    weight_pos=jnp.ones_like(feature_pos[:, :, 0]),
                    weight_neg=repeat(uncond_w, 'b -> (b f) k', f=B // uncond_w.shape[0], k=n_uncond),
                    **loss_kwargs,
                )
                return loss, info
            
            loss_per_feature = jax.tree.map(feature_loss, sg_features, gen_features)
            total_loss = 0
            total_info = dict()
            for k, v in loss_per_feature.items():
                total_loss = total_loss + v[0].mean()
                for k2, v2 in v[1].items():
                    total_info[f'{k2}/{k}'] = v2
            total_loss = total_loss.mean()
            total_info = jax.tree.map(lambda x: x.mean(), total_info)

            return total_loss, total_info

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metric), grads = grad_fn(state.params)
        return loss, metric, grads

    loss, metric, grads = loss_grad_info(labels, samples, negative_samples, cfg, rng_step)

    g_norm = optax.global_norm(grads)
    clipper = optax.clip_by_global_norm(max_grad_norm)
    updates, _ = clipper.update(grads, None)
    
    new_state = state.apply_gradients(grads=updates)

    new_ema_params = jax.tree.map(
        lambda ema, p: ema * state.ema_decay + p * (1.0 - state.ema_decay),
        state.ema_params,
        new_state.params,
    )
    new_state = new_state.replace(ema_params=new_ema_params)
    
    metric['loss'] = loss
    metric['g_norm'] = g_norm
    metric['lr'] = learning_rate_fn(state.step)
    metric = jax.tree.map(lambda x: x.mean(), metric)
    return new_state, metric

def generate_step(batch, params, rng, apply_fn, postprocess_fn, cfg_scale=1.0):
    """Generate samples from a batch of labels for FID evaluation.

    Args:
        batch: tuple ``(images, labels)`` from ``epoch0_sampler``; only labels are used.
        params: generator parameter tree.
        rng: PRNGKey for noise sampling.
        apply_fn: model ``apply`` callable.
        postprocess_fn: maps raw model output ``(B, H, W, C)`` to uint8 ``(B, C, H, W)`` or ``(B, H, W, C)``.
        cfg_scale: classifier-free guidance scale.

    Returns:
        Postprocessed samples with shape ``(B, ...)``.
    """
    _, labels = batch
    labels = jax.lax.with_sharding_constraint(labels, data_shard())
    latent_samples = apply_fn(
        {'params': params},
        train=False,
        rngs=prepare_rng(rng, ['noise']),
        c=labels,
        cfg_scale=cfg_scale,
    )['samples']
    latent_samples = jax.tree_util.tree_map(
        lambda x: jax.lax.with_sharding_constraint(x, ddp_shard()),
        latent_samples
    )
    return postprocess_fn(latent_samples)
def train_gen(
    model,  # DitGen model instance
    optimizer,  # Optax optimizer transform
    logger,  # logger with log_dict / finish
    eval_loader,  # evaluation dataloader iterator source
    train_loader,  # training dataloader iterator source
    learning_rate_fn,  # callable(step) -> lr
    preprocess_fn,  # preprocessing function for dataloader batches
    postprocess_fn,  # generated sample postprocess function
    dataset_name="imagenet256",  # dataset name for eval logging
    train_batch_size=0,  # override per-host train batch if > 0
    total_steps=100000,  # max optimization steps
    save_per_step=10000,  # checkpoint save interval
    eval_per_step=5000,  # evaluation interval
    eval_samples=50000,  # number of generated samples for FID evaluation
    activation_fn=None,  # feature function used by drift loss
    feature_params=None,  # params bundle consumed by activation_fn
    ema_decay=0.999,  # single EMA decay
    seed=42,  # global RNG seed
    pos_per_sample=32,  # positive samples from memory bank
    neg_per_sample=16,  # negative samples from memory bank
    forward_dict=dict(
        gen_per_label=16,
        cfg_min=1.0,
        cfg_max=4.0,
        neg_cfg_pw=1.0,
        no_cfg_frac=0.0,
    ), 
    positive_bank_size=64,
    negative_bank_size=512,
    cfg_list=(1.0,),
    activation_kwargs=dict(
        patch_mean_size=[2,4],
        patch_std_size=[2,4],
        use_std=True,
        use_mean=True,
        every_k_block=2,
    ),
    max_grad_norm=2.0,
    loss_kwargs=dict(R_list=(0.02, 0.05, 0.2)),
    keep_every=500000,  # long-term checkpoint retention interval
    keep_last=2,  # number of latest checkpoints to keep
    init_from="",  # `hf://<name>` or local dir of model
    push_per_step=0,  # memory-bank fill factor per train step
    push_at_resume=3000,  # extra fill multiplier when resuming
    workdir="runs",  # run root containing checkpoints/logs
):
    """
    Main training loop.
    """
    if isinstance(ema_decay, (list, tuple)):
        if len(ema_decay) != 1:
            raise ValueError(f"Expected a single ema_decay value, got {ema_decay}")
        ema_decay = float(ema_decay[0])
    else:
        ema_decay = float(ema_decay)
    if cfg_list is None:
        cfg_list = [1.0]
    elif isinstance(cfg_list, (list, tuple)):
        cfg_list = [float(cfg) for cfg in cfg_list]
    else:
        cfg_list = [float(cfg_list)]

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    rng_train, rng_eval = jax.random.split(rng)
    state = init_state_from_dummy_input(model, optimizer, TrainState, rng, model.dummy_input(), model.rng_keys(), ema_decay=ema_decay)
    state = restore_checkpoint(state=state, workdir=workdir)
    if int(jax.device_get(state.step)) == 0 and init_from:
        log_for_0("Initializing generator params from init_from=%s", init_from)
        state = maybe_init_state_params(
            state,
            model_type="generator",
            init_from=init_from,
            hf_cache_dir=HF_ROOT,
        )
    gen_step_jit = jax.jit(partial(generate_step, apply_fn=state.apply_fn, postprocess_fn=postprocess_fn))
    assert feature_params is not None, "feature_params must be provided for multi-host safe feature extraction"
    loss_kwargs['R_list'] = tuple(loss_kwargs['R_list'])
    state_sharding = jax.tree.map(lambda x: x.sharding, state)
    train_step_jit = jax.jit(partial(train_step, rng_init=rng_train, learning_rate_fn=learning_rate_fn, feature_apply=activation_fn, activation_kwargs=activation_kwargs, loss_kwargs=loss_kwargs, **forward_dict, max_grad_norm=max_grad_norm), out_shardings=(state_sharding, None))

    ema_to_params_func = map_to_sharding(state.params)
    
    log_for_0("Starting training loop...")
    step = int(state.step)
    initial_step = step
    pbar = tqdm(range(step, total_steps), initial=step, total=total_steps) if is_rank_zero() else range(step, total_steps)
    memory_bank_positive = ArrayMemoryBank(num_classes=1000, max_size=positive_bank_size)
    memory_bank_negative = ArrayMemoryBank(num_classes=1, max_size=negative_bank_size)
    mu.sync_global_devices("train loop started")
    train_iter = infinite_sampler(train_loader, step)

    print(f"process_count={jax.process_count()} "
            f"local_device_count={jax.local_device_count()} "
            f"device_count={jax.device_count()}")

    for step in pbar:
        start_time = time.time()
        n_push = 0
        logger.set_step(step)

        # do push to memory bank; per host 
        goal = push_per_step
        if initial_step > 0 and step == initial_step:
            goal = push_at_resume * push_per_step
            print(f"pushing at resume: {goal}")
        while True:
            batch = next(train_iter)
            # Preprocess batch: converts (images, labels) tuple to {'images': BHWC, 'labels': ...}
            processed_batch = preprocess_fn(batch)
            images = processed_batch['images']  # BHWC format
            labels = processed_batch['labels']
            memory_bank_positive.add(images, labels)
            memory_bank_negative.add(images, labels * 0)
            n_push += images.shape[0]
            if n_push >= goal:
                break
        
        bsz_per_host = train_batch_size // jax.process_count()
        assert labels.shape[0] >= bsz_per_host, f"Labels shape {labels.shape[0]} < bsz_per_host {bsz_per_host}"
        select_indices = jax.random.choice(jax.random.fold_in(rng_train, step), jnp.arange(labels.shape[0]), (bsz_per_host,), replace=False)
        labels = labels[select_indices]
        images = images[select_indices]

        positive_samples = memory_bank_positive.sample(labels, n_samples=pos_per_sample)
        negative_samples = memory_bank_negative.sample(labels * 0, n_samples=neg_per_sample)

        merged_positive, merged_negative, merged_labels = merge_data((positive_samples, negative_samples, labels))

        process_time = time.time() - start_time

        profile_metrics = dict()
        if (step == initial_step):
            profile_metrics = profile_func(train_step_jit, (state, merged_labels, merged_positive, merged_negative, feature_params), name="train_step")

        new_state, metrics = train_step_jit(state, merged_labels, merged_positive, merged_negative, feature_params)
        metrics = jax.tree.map(lambda x: x.mean(), metrics)
        total_time = time.time() - start_time
        metrics['total_time'] = total_time
        metrics['process_time'] = process_time
        metrics['kimg'] = (step + 1) * merged_positive.shape[0] / 1000.0
        metrics['forward_kimg'] = (step + 1) * merged_positive.shape[0] / 1000.0 * forward_dict['gen_per_label']
        metrics.update(profile_metrics)
    
        logger.log_dict(metrics)
        state = new_state
        step += 1

        if step % save_per_step == 0 or step == total_steps: 
            mu.sync_global_devices("save checkpoint started")
            save_checkpoint(state, keep=keep_last, keep_every=keep_every, workdir=workdir)
            save_params_ema_artifact(
                state,
                workdir=workdir,
                kind="gen",
                model_config=_generator_model_config(model),
            )
            mu.sync_global_devices("save checkpoint finished")

        if (step % eval_per_step == 0) or (step == 1) or (step == total_steps):
            is_sanity = (step == 1)  # do a sanity check, to make sure FID env is working

            n_samples = 500 if is_sanity else eval_samples
            folder_prefix = "sanity" if is_sanity else "CFG"
            eval_params = ema_to_params_func(state.ema_params)
            round_best_fid = float("inf")
            round_best_cfg = cfg_list[0]
            eval_cfg_list = cfg_list if not is_sanity else [cfg_list[0]]

            for eval_cfg in eval_cfg_list:
                mu.sync_global_devices("eval started")
                result = evaluate_fid(
                    dataset_name=dataset_name,
                    gen_func=gen_step_jit,
                    gen_params={"params": eval_params, "cfg_scale": eval_cfg},
                    eval_loader=eval_loader,
                    logger=logger,
                    num_samples=n_samples,
                    log_folder=f"{folder_prefix}{eval_cfg}",
                    log_prefix=f"EMA_{state.ema_decay:g}",
                    rng_eval=rng_eval,
                )
                mu.sync_global_devices("eval finished")
                fid_val = result.get("fid", float("inf"))
                if fid_val < round_best_fid:
                    round_best_fid = fid_val
                    round_best_cfg = eval_cfg
            if not is_sanity:
                log_for_0("best_fid=%.4f best_cfg=%.1f (step=%d)", round_best_fid, round_best_cfg, step)
                logger.log_dict({"best_fid": round_best_fid, "best_cfg": round_best_cfg})

        if step % 100 == 0:
            mu.sync_global_devices(f"train step {step} finished")


    mu.sync_global_devices("train loop finished")
    logger.finish()
    del model, optimizer, eval_loader, train_loader, state    
    gc.collect()
    jax.clear_caches()
    mu.sync_global_devices("train loop finished")


def main_gen(config, output_dir="runs"):
    if "logging" not in config:
        config.logging = {}
    config.logging.name = Path(output_dir).resolve().name
        
    from models.generator import DitGen
    
    set_global_mesh(config.get("hsdp_dim", min(8, jax.local_device_count() * jax.process_count())))
    
    model_dict = build_model_dict(config, DitGen, workdir=output_dir)
    use_aug = bool(config.dataset.get("use_aug", False))
    use_latent = bool(config.dataset.get("use_latent", False))
    use_cache = bool(config.dataset.get("use_cache", False))
    postprocess_fn_noclip = get_postprocess_fn(
        use_aug=use_aug,
        use_latent=use_latent,
        use_cache=use_cache,
        has_clip=False,
    )
    feature_cfg = model_dict.feature
    mae_path = str(feature_cfg.get("mae_path", "")).strip()
    if not mae_path and bool(feature_cfg.get("use_mae", True)):
        load_dict = feature_cfg.get("load_dict", {})
        if str(load_dict.get("source", "hf")).strip().lower() == "local":
            mae_path = str(load_dict.get("path", "")).strip()
        else:
            model_name = str(load_dict.get("hf_model_name", "")).strip()
            if model_name:
                mae_path = f"hf://{model_name}"
    if bool(feature_cfg.get("use_mae", True)) and not mae_path:
        raise ValueError("feature.mae_path (or feature.load_dict.hf_model_name / feature.load_dict.path) is required when use_mae=true.")
    activation_fn, variables = build_activation_function(
        mae_path=mae_path,
        use_convnext=bool(feature_cfg.get("use_convnext", False)),
        convnext_bf16=bool(feature_cfg.get("convnext_bf16", False)),
        use_mae=bool(feature_cfg.get("use_mae", True)),
        postprocess_fn=postprocess_fn_noclip,
    )
    train_gen(
        model=model_dict.model,
        optimizer=model_dict.optimizer,
        logger=model_dict.logger,
        eval_loader=model_dict.eval_loader,
        train_loader=model_dict.train_loader,
        learning_rate_fn=model_dict.learning_rate_fn,
        preprocess_fn=model_dict.preprocess_fn,
        postprocess_fn=model_dict.postprocess_fn,
        dataset_name=model_dict.dataset_name,
        activation_fn=activation_fn,
        feature_params=variables,
        workdir=output_dir,
        **config.train
    )
    mu.sync_global_devices("main_gen finished")
    del model_dict
    gc.collect()
    jax.clear_caches()
    mu.sync_global_devices("main_gen finished")

def main(args):
    run_init()
    config = load_config(args.config)
    main_gen(config, output_dir=args.workdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gen/latent_ablation.yaml", help="Path to configuration file.")
    parser.add_argument("--workdir", type=str, default="runs", help="Local workdir root for checkpoints/logs.")
    args = parser.parse_args()
    args.output_dir = args.workdir

    main(args)
