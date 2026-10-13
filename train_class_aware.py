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

def cdist(x, y, eps=1e-8):
    xydot = jnp.einsum("bnd,bmd->bnm", x, y)
    xnorms = jnp.einsum("bnd,bnd->bn", x, x)
    ynorms = jnp.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return jnp.sqrt(jnp.clip(sq_dist, a_min=eps))

@partial(jax.jit, static_argnames=("R_list", "mask_self", "gamma_a_same", "gamma_a_diff", "gamma_r_same", "gamma_r_diff"))
def drift_loss_class_aware(
    gen,
    bank_features,
    y_neg,
    labels_gen,
    labels_bank,
    labels_neg,
    R_list=(0.02, 0.05, 0.2),
    gamma_a_same=1.5,
    gamma_a_diff=0.75,
    gamma_r_same=1.0,
    gamma_r_diff=1.5,
    mask_self=True
):
    B, C_g, S = gen.shape
    C_b = bank_features.shape[1]
    
    old_gen = jax.lax.stop_gradient(gen)
    y_neg = jax.lax.stop_gradient(y_neg)
    
    targets = jnp.concatenate([y_neg, bank_features], axis=1) # [B, C_g + C_b, S]
    targets_w = jnp.ones_like(targets[:, :, 0])
    
    mask_same_neg = (labels_gen[:, :, None] == labels_neg[:, None, :]).astype(jnp.float32)
    mask_diff_neg = 1.0 - mask_same_neg

    mask_same_pos = (labels_gen[:, :, None] == labels_bank[:, None, :]).astype(jnp.float32)
    mask_diff_pos = 1.0 - mask_same_pos

    attraction_weight = mask_same_pos * gamma_a_same + mask_diff_pos * gamma_a_diff
    repulsion_weight = mask_same_neg * gamma_r_same + mask_diff_neg * gamma_r_diff

    def calculate_scaled_goal_and_factor(old_gen_in, targets_in, targets_w_in):
        info = {}
        dist = cdist(old_gen_in, targets_in)
        weighted_dist = dist * targets_w_in[:, None, :]
        scale = weighted_dist.mean() / targets_w_in.mean()
        info["scale"] = scale

        scale_inputs = jnp.clip(scale / jnp.sqrt(S), a_min=1e-3)
        old_gen_scaled = old_gen_in / scale_inputs
        targets_scaled = targets_in / scale_inputs
        dist_normed = dist / jnp.clip(scale, a_min=1e-3)
        
        mask_val = 100.0
        diag_mask = jnp.eye(C_g, dtype=jnp.float32)
        split_idx = C_g // 2
        diag_mask = diag_mask.at[split_idx:, split_idx:].set(0.0)
        block_mask = jnp.pad(diag_mask, ((0, 0), (0, C_b))) 
        dist_normed = dist_normed + jnp.expand_dims(block_mask, 0) * mask_val

        force_across_R = jnp.zeros_like(old_gen_scaled)
        
        for R in R_list:
            logits = -dist_normed / R
            affinity = jax.nn.softmax(logits, axis=-1)
            aff_transpose = jax.nn.softmax(logits, axis=-2)
            affinity = jnp.sqrt(jnp.clip(affinity * aff_transpose, a_min=1e-6))
            affinity = affinity * targets_w_in[:, None, :]

            A_neg = affinity[:, :, :C_g] # [B, C_g, C_g]
            A_pos = affinity[:, :, C_g:] # [B, C_g, C_b]

            W_pos = A_pos * attraction_weight
            W_neg = A_neg * repulsion_weight
            
            sum_W_neg = jnp.sum(W_neg, axis=-1, keepdims=True)
            sum_W_pos = jnp.sum(W_pos, axis=-1, keepdims=True)
            
            r_coeff_pos = W_pos * sum_W_neg
            r_coeff_neg = - W_neg * sum_W_pos 
            
            R_coeff = jnp.concatenate([r_coeff_neg, r_coeff_pos], axis=2)
            total_force_R = jnp.einsum("biy,byx->bix", R_coeff, targets_scaled)

            total_coeffs = R_coeff.sum(axis=-1)
            total_force_R = total_force_R - total_coeffs[..., None] * old_gen_scaled
            f_norm_val = (total_force_R ** 2).mean()

            info[f"loss_{R}"] = f_norm_val

            force_scale = jnp.sqrt(jnp.clip(f_norm_val, a_min=1e-8))
            force_across_R = force_across_R + total_force_R / force_scale

        goal_scaled = old_gen_scaled + force_across_R
        return goal_scaled, scale_inputs, info

    goal_scaled, scale_inputs, info = jax.lax.stop_gradient(
        calculate_scaled_goal_and_factor(old_gen, targets, targets_w)
    )
    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    loss = jnp.mean(diff ** 2, axis=(-1, -2))
    info = jax.tree.map(lambda x: x.mean(), info)

    return loss, info

class TrainState(train_state.TrainState):
    ema_params: Optional[Any] = None
    ema_decay: float = 0.999

def _generator_model_config(model) -> dict:
    return {
        name: value
        for name, value in vars(model).items()
        if name not in {"parent", "name"} and not name.startswith("_")
    }

def train_step(state: TrainState, labels, bank_samples, bank_labels, feature_params, feature_apply, rng_init, learning_rate_fn=None, gen_per_label=8, activation_kwargs=dict(), loss_kwargs=dict(R_list=[0.02, 0.05, 0.2]), max_grad_norm=2.0):
    rng_step = jax.random.fold_in(rng_init, state.step)

    def loss_grad_info(labels, bank_samples, bank_labels, rng_step):
        labels = enforce_ddp(labels)
        bank_samples = enforce_ddp(bank_samples)
        bank_labels = enforce_ddp(bank_labels)
        bsz = labels.shape[0]
        
        n_bank = bank_samples.shape[1]
        n_gen = gen_per_label
        bank_samples_input = rearrange(bank_samples, 'b x ... -> (b x) ...')
        bank_samples_input = enforce_ddp(bank_samples_input)
        
        sg_features = jax.lax.stop_gradient(feature_apply(feature_params, bank_samples_input, **activation_kwargs))
        if bsz % jax.device_count() == 0:
            sg_features = jax.tree.map(lambda u: rearrange(u, '(b x) ... -> b x ...', x=n_bank), sg_features) 
        else:
            sg_features = jax.tree.map(lambda u: rearrange(enforce_ddp(u), '(b x) ... -> b x ...', x=n_bank), sg_features) 
        sg_features = enforce_ddp(sg_features)

        def loss_fn(params):
            input_labels = enforce_ddp(repeat(labels, 'b -> (b g)', g=gen_per_label))
            
            gen_samples = state.apply_fn(
                {'params': params},
                train=True,
                rngs=prepare_rng(rng_step, ['noise']),
                c=input_labels,
                cfg_scale=1.0,
            )['samples'] 
            
            gen_features = feature_apply(feature_params, gen_samples, **activation_kwargs)
            if bsz % jax.device_count() == 0:
                gen_features = jax.tree.map(lambda u: rearrange(u, '(b g) ... -> b g ...', g=n_gen), gen_features)
            else:
                gen_features = jax.tree.map(lambda u: rearrange(enforce_ddp(u), '(b g) ... -> b g ...', g=n_gen), gen_features)
            gen_features = enforce_ddp(gen_features)

            def feature_loss(sg_feat, gen_feat):
                bsz = gen_feat.shape[0]
                n_gen = gen_feat.shape[1]
                n_patches = gen_feat.shape[2]
                split_idx = n_gen // 2
                
                gen_same = gen_feat[:, :split_idx]
                idx = (jnp.arange(bsz) + bsz // 2) % bsz
                gen_diff = gen_feat[idx, split_idx:]
                gen_combined = jnp.concatenate([gen_same, gen_diff], axis=1)
                
                labels_same = repeat(labels, 'b -> b g', g=split_idx)
                labels_diff = repeat(labels[idx], 'b -> b g', g=split_idx)
                labels_gen_combined = jnp.concatenate([labels_same, labels_diff], axis=1)
                labels_x_2d = repeat(labels, 'b -> b g', g=n_gen)
                
                bank_feat_folded = enforce_ddp(rearrange(sg_feat, 'b x f d -> (b f) x d'))
                gen_feat_folded = enforce_ddp(rearrange(gen_feat, 'b x f d -> (b f) x d'))
                gen_combined_folded = enforce_ddp(rearrange(gen_combined, 'b x f d -> (b f) x d'))
                
                labels_x_folded = repeat(labels_x_2d, 'b g -> (b f) g', f=n_patches)
                labels_gen_combined_folded = repeat(labels_gen_combined, 'b g -> (b f) g', f=n_patches)
                labels_bank_folded = repeat(bank_labels, 'b x -> (b f) x', f=n_patches)
                
                loss, info = drift_loss_class_aware(
                    gen=gen_feat_folded,
                    bank_features=bank_feat_folded,
                    y_neg=gen_combined_folded,
                    labels_gen=labels_x_folded,
                    labels_bank=labels_bank_folded,
                    labels_neg=labels_gen_combined_folded,
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

    loss, metric, grads = loss_grad_info(labels, bank_samples, bank_labels, rng_step)

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
    if learning_rate_fn: metric['lr'] = learning_rate_fn(state.step)
    metric = jax.tree.map(lambda x: x.mean(), metric)
    return new_state, metric

def generate_step(batch, params, rng, apply_fn, postprocess_fn, cfg_scale=1.0):
    _, labels = batch
    labels = jax.lax.with_sharding_constraint(labels, data_shard())
    latent_samples = apply_fn(
        {'params': params},
        train=False,
        rngs=prepare_rng(rng, ['noise']),
        c=labels,
        cfg_scale=cfg_scale, # 统一标量传入
    )['samples']
    latent_samples = jax.tree_util.tree_map(
        lambda x: jax.lax.with_sharding_constraint(x, ddp_shard()),
        latent_samples
    )
    return postprocess_fn(latent_samples)

def train_gen(
    model, optimizer, logger, eval_loader, train_loader, learning_rate_fn, preprocess_fn, postprocess_fn,
    dataset_name="imagenet256", train_batch_size=0, total_steps=100000, save_per_step=10000, eval_per_step=5000, eval_samples=50000,
    activation_fn=None, feature_params=None, ema_decay=0.999, seed=42, 
    pos_per_sample=16, neg_per_sample=16, 
    forward_dict=dict(gen_per_label=32), 
    bank_size=512,
    cfg_list=(1.0,),
    activation_kwargs=dict(patch_mean_size=[2,4], patch_std_size=[2,4], use_std=True, use_mean=True, every_k_block=2),
    max_grad_norm=2.0, loss_kwargs=dict(R_list=(0.02, 0.05, 0.2)),
    keep_every=500000, keep_last=2, init_from="", push_per_step=0, push_at_resume=3000, workdir="runs",
):
    if isinstance(ema_decay, (list, tuple)): ema_decay = float(ema_decay[0])
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    rng_train, rng_eval = jax.random.split(rng)
    state = init_state_from_dummy_input(model, optimizer, TrainState, rng, model.dummy_input(), model.rng_keys(), ema_decay=ema_decay)
    state = restore_checkpoint(state=state, workdir=workdir)
    if int(jax.device_get(state.step)) == 0 and init_from:
        state = maybe_init_state_params(state, model_type="generator", init_from=init_from, hf_cache_dir=HF_ROOT)
    
    gen_step_jit = jax.jit(partial(generate_step, apply_fn=state.apply_fn, postprocess_fn=postprocess_fn))
    loss_kwargs['R_list'] = tuple(loss_kwargs['R_list'])
    safe_loss_kwargs = {k: v for k, v in loss_kwargs.items() if k in ("R_list", "gamma_a_same", "gamma_a_diff", "gamma_r_same", "gamma_r_diff")}
    
    state_sharding = jax.tree.map(lambda x: x.sharding, state)
    train_step_jit = jax.jit(partial(train_step, rng_init=rng_train, learning_rate_fn=learning_rate_fn, feature_apply=activation_fn, activation_kwargs=activation_kwargs, loss_kwargs=safe_loss_kwargs, **forward_dict, max_grad_norm=max_grad_norm), out_shardings=(state_sharding, None))
    ema_to_params_func = map_to_sharding(state.params)
    
    step = int(state.step)
    initial_step = step
    pbar = tqdm(range(step, total_steps), initial=step, total=total_steps) if is_rank_zero() else range(step, total_steps)
    
    memory_bank_all = ArrayMemoryBank(num_classes=1000, max_size=bank_size)
    
    mu.sync_global_devices("train loop started")
    train_iter = infinite_sampler(train_loader, step)

    for step in pbar:
        start_time = time.time()
        n_push = 0
        logger.set_step(step)

        goal = push_per_step
        if initial_step > 0 and step == initial_step: goal = push_at_resume * push_per_step
        while True:
            batch = next(train_iter)
            processed_batch = preprocess_fn(batch)
            images = processed_batch['images']
            labels = processed_batch['labels']
            memory_bank_all.add(images, labels)
            n_push += images.shape[0]
            if n_push >= goal: break
        
        bsz_per_host = train_batch_size // jax.process_count()
        select_indices = jax.random.choice(jax.random.fold_in(rng_train, step), jnp.arange(labels.shape[0]), (bsz_per_host,), replace=False)
        labels = labels[select_indices]
        images = images[select_indices]

        positive_samples = memory_bank_all.sample(labels, n_samples=pos_per_sample) 
        pos_labels = repeat(labels, 'b -> b p', p=pos_per_sample)
        
        rng_rand = jax.random.fold_in(rng_train, step + 10086)
        rand_labels = jax.random.randint(rng_rand, (bsz_per_host,), 0, 1000)
        negative_samples = memory_bank_all.sample(rand_labels, n_samples=neg_per_sample)
        neg_labels = repeat(rand_labels, 'b -> b n', n=neg_per_sample) 
        
        bank_samples = jnp.concatenate([positive_samples, negative_samples], axis=1)
        bank_labels = jnp.concatenate([pos_labels, neg_labels], axis=1)

        merged_sample, merged_bank, merged_bank_labels, merged_labels = merge_data((images, bank_samples, bank_labels, labels))

        process_time = time.time() - start_time
        new_state, metrics = train_step_jit(state, merged_labels, merged_bank, merged_bank_labels, feature_params)
        metrics = jax.tree.map(lambda x: x.mean(), metrics)
        
        total_time = time.time() - start_time
        metrics['total_time'] = total_time
        metrics['process_time'] = process_time
        metrics['kimg'] = (step + 1) * merged_sample.shape[0] / 1000.0
        logger.log_dict(metrics)
        state = new_state
        step += 1

        if step % save_per_step == 0 or step == total_steps: 
            mu.sync_global_devices("save checkpoint started")
            save_checkpoint(state, keep=keep_last, keep_every=keep_every, workdir=workdir)
            save_params_ema_artifact(state, workdir=workdir, kind="gen", model_config=_generator_model_config(model))
            mu.sync_global_devices("save checkpoint finished")

        if (step % eval_per_step == 0) or (step == 1) or (step == total_steps):
            is_sanity = (step == 1)
            n_samples = 500 if is_sanity else eval_samples
            folder_prefix = "sanity" if is_sanity else "CFG"
            eval_params = ema_to_params_func(state.ema_params)
            
            mu.sync_global_devices("eval started")
            result = evaluate_fid(
                dataset_name=dataset_name, gen_func=gen_step_jit,
                gen_params={"params": eval_params, "cfg_scale": 1.0},
                eval_loader=eval_loader, logger=logger, num_samples=n_samples,
                log_folder=f"{folder_prefix}1.0", log_prefix=f"EMA_{state.ema_decay:g}", rng_eval=rng_eval,
            )
            mu.sync_global_devices("eval finished")
            fid_val = result.get("fid", float("inf"))
            if not is_sanity:
                log_for_0("best_fid=%.4f (step=%d)", fid_val, step)
                logger.log_dict({"best_fid": fid_val})

    mu.sync_global_devices("train loop finished")
    logger.finish()
    del model, optimizer, eval_loader, train_loader, state    
    gc.collect()
    jax.clear_caches()
    mu.sync_global_devices("train loop finished")

def main_gen(config, output_dir="runs"):
    if "logging" not in config: config.logging = {}
    config.logging.name = Path(output_dir).resolve().name
    from models.generator import DitGen
    set_global_mesh(config.get("hsdp_dim", min(8, jax.local_device_count() * jax.process_count())))
    model_dict = build_model_dict(config, DitGen, workdir=output_dir)
    use_aug = bool(config.dataset.get("use_aug", False))
    use_latent = bool(config.dataset.get("use_latent", False))
    use_cache = bool(config.dataset.get("use_cache", False))
    postprocess_fn_noclip = get_postprocess_fn(use_aug=use_aug, use_latent=use_latent, use_cache=use_cache, has_clip=False)
    feature_cfg = model_dict.feature
    mae_path = str(feature_cfg.get("mae_path", "")).strip()
    if not mae_path and bool(feature_cfg.get("use_mae", True)):
        load_dict = feature_cfg.get("load_dict", {})
        if str(load_dict.get("source", "hf")).strip().lower() == "local":
            mae_path = str(load_dict.get("path", "")).strip()
        else:
            model_name = str(load_dict.get("hf_model_name", "")).strip()
            if model_name: mae_path = f"hf://{model_name}"
    activation_fn, variables = build_activation_function(
        mae_path=mae_path, use_convnext=bool(feature_cfg.get("use_convnext", False)),
        convnext_bf16=bool(feature_cfg.get("convnext_bf16", False)),
        use_mae=bool(feature_cfg.get("use_mae", True)), postprocess_fn=postprocess_fn_noclip,
    )
    train_gen(
        model=model_dict.model, optimizer=model_dict.optimizer, logger=model_dict.logger,
        eval_loader=model_dict.eval_loader, train_loader=model_dict.train_loader,
        learning_rate_fn=model_dict.learning_rate_fn, preprocess_fn=model_dict.preprocess_fn,
        postprocess_fn=model_dict.postprocess_fn, dataset_name=model_dict.dataset_name,
        activation_fn=activation_fn, feature_params=variables, workdir=output_dir, **config.train
    )

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
