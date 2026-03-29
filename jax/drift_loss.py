import jax
import jax.numpy as jnp
from functools import partial
from einops import repeat

def cdist(x, y, eps=1e-8):
    # [B, N, D] x [B, M, D] -> [B, N, M]
    xydot = jnp.einsum("bnd,bmd->bnm", x, y)
    xnorms = jnp.einsum("bnd,bnd->bn", x, x)
    ynorms = jnp.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return jnp.sqrt(jnp.clip(sq_dist, a_min=eps))

@partial(jax.jit, static_argnames=("R_list",))
def drift_loss(
    gen,
    fixed_pos,
    fixed_neg=None,
    weight_gen=None,
    weight_pos=None,
    weight_neg=None,
    R_list=(0.02, 0.05, 0.2),
):
    '''
    Args:
        gen: [B, C_g, S]
        fixed_pos: [B, C_p, S]
        fixed_neg: [B, C_n, S] (optional, can be None)
        weight_gen: [B, C_g] (optional; if None: weight is 1)
        weight_pos: [B, C_p] (optional; if None: weight is 1)
        weight_neg: [B, C_n] (optional; if None: weight is 1)
        R_list: a list of R values to use for the kernel function
    Returns:
        loss: [batch_size]
        (optional) info: a dict with entries:
            scale: the scale of the loss 
            loss_R: the loss for each R value
    '''
    
    # 1. Defaults & Casting
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]
    
    if fixed_neg is None:
        fixed_neg = jnp.zeros_like(gen[:, :0, :])
    C_n = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = jnp.ones_like(gen[:, :, 0])
    if weight_pos is None:
        weight_pos = jnp.ones_like(fixed_pos[:, :, 0])
    if weight_neg is None:
        weight_neg = jnp.ones_like(fixed_neg[:, :, 0])
    gen = gen.astype(jnp.float32)
    fixed_pos = fixed_pos.astype(jnp.float32)
    fixed_neg = fixed_neg.astype(jnp.float32)
    weight_gen = weight_gen.astype(jnp.float32)
    weight_pos = weight_pos.astype(jnp.float32)
    weight_neg = weight_neg.astype(jnp.float32)
    old_gen = jax.lax.stop_gradient(gen)
    targets = jnp.concatenate([old_gen, fixed_neg, fixed_pos], axis=1)
    targets_w = jnp.concatenate([weight_gen, weight_neg, weight_pos], axis=1)

    # 2. Core Logic (Wrapped for stop_gradient)
    def calculate_scaled_goal_and_factor(old_gen_in, targets_in, targets_w_in):
        # --- Scaling ---
        info = {}
        dist = cdist(old_gen_in, targets_in)
        weighted_dist = dist * targets_w_in[:, None, :] # [B, C_g, C_g + C_n + C_p]
        scale = weighted_dist.mean() / targets_w_in.mean() # [B]
        info["scale"] = scale

        scale_inputs = jnp.clip(scale / jnp.sqrt(S), a_min=1e-3) # Normalize coords to have order 1
        old_gen_scaled = old_gen_in / scale_inputs
        targets_scaled = targets_in / scale_inputs
        
        # Normalize distance for kernel
        dist_normed = dist / jnp.clip(scale, a_min=1e-3)
        
        # --- Masking ---
        mask_val = 100.0
        diag_mask = jnp.eye(C_g, dtype=jnp.float32)
        block_mask = jnp.pad(diag_mask, ((0, 0), (0, C_n + C_p))) 
        block_mask = jnp.expand_dims(block_mask, 0)
        dist_normed = dist_normed + block_mask * mask_val

        # --- Force Loop ---
        force_across_R = jnp.zeros_like(old_gen_scaled)
        
        for R in R_list:
            logits = -dist_normed / R

            affinity = jax.nn.softmax(logits, axis=-1)
            aff_transpose = jax.nn.softmax(logits, axis=-2)
            affinity = jnp.sqrt(jnp.clip(affinity * aff_transpose, a_min=1e-6))

            affinity = affinity * targets_w_in[:, None, :]

            split_idx = C_g + C_n
            aff_neg = affinity[:, :, :split_idx]
            aff_pos = affinity[:, :, split_idx:]
            
            sum_pos = jnp.sum(aff_pos, axis=-1, keepdims=True)
            r_coeff_neg = -aff_neg * sum_pos 
            sum_neg = jnp.sum(aff_neg, axis=-1, keepdims=True)
            r_coeff_pos = aff_pos * sum_neg 
            
            R_coeff = jnp.concatenate([r_coeff_neg, r_coeff_pos], axis=2)

            total_force_R = jnp.einsum("biy,byx->bix", R_coeff, targets_scaled)

            total_coeffs = R_coeff.sum(axis=-1) # guaranteed to be 0, in no_repulsion case
            total_force_R = total_force_R - total_coeffs[..., None] * old_gen_scaled
            f_norm_val = (total_force_R ** 2).mean() # [B]

            info[f"loss_{R}"] = f_norm_val

            force_scale = jnp.sqrt(jnp.clip(f_norm_val, a_min=1e-8)) # normalize force of each temperature
            force_across_R = force_across_R + total_force_R / force_scale

        goal_scaled = old_gen_scaled + force_across_R
        
        return goal_scaled, scale_inputs, info

    # 3. Compute Goal (No Gradients)
    goal_scaled, scale_inputs, info = jax.lax.stop_gradient(
        calculate_scaled_goal_and_factor(old_gen, targets, targets_w)
    )
    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    loss = jnp.mean(diff ** 2, axis=(-1, -2))
    info = jax.tree.map(lambda x: x.mean(), info)

    return loss, info
