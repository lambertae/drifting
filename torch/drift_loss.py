"""PyTorch implementation of the Drift loss."""
from __future__ import annotations
from typing import Optional, Tuple, Dict, List
import torch
import torch.nn.functional as F


def cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Batched pairwise Euclidean distance.

    Args:
        x: [B, N, D]
        y: [B, M, D]
    Returns:
        distances: [B, N, M]
    """
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return torch.sqrt(sq_dist.clamp(min=eps))


def drift_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: Optional[torch.Tensor] = None,
    weight_gen: Optional[torch.Tensor] = None,
    weight_pos: Optional[torch.Tensor] = None,
    weight_neg: Optional[torch.Tensor] = None,
    R_list: Tuple[float, ...] = (0.02, 0.05, 0.2),
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the Drift loss.

    Args:
        gen:       Generated tokens  [B, C_g, S]
        fixed_pos: Positive anchors  [B, C_p, S]
        fixed_neg: Negative anchors  [B, C_n, S] or None
        weight_gen, weight_pos, weight_neg: per-token scalar weights [B, C_*]
        R_list: bandwidth radii for computing attraction/repulsion forces

    Returns:
        loss: per-sample scalar loss [B]
        info: dict of diagnostic scalars
    """
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]

    if fixed_neg is None:
        fixed_neg = gen.new_zeros(B, 0, S)
    C_n = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = gen.new_ones(B, C_g)
    if weight_pos is None:
        weight_pos = gen.new_ones(B, C_p)
    if weight_neg is None:
        weight_neg = gen.new_ones(B, C_n)

    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    old_gen = gen.detach()

    # targets: [B, C_g + C_n + C_p, S]
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    def calculate_scaled_goal_and_factor(
        old_gen_in: torch.Tensor,
        targets_in: torch.Tensor,
        targets_w_in: torch.Tensor,
    ):
        info: Dict[str, torch.Tensor] = {}

        dist = cdist(old_gen_in, targets_in)           # [B, C_g, C_g+C_n+C_p]
        weighted_dist = dist * targets_w_in[:, None, :]
        scale = weighted_dist.mean() / targets_w_in.mean()
        info["scale"] = scale

        scale_inputs = (scale / (S ** 0.5)).clamp(min=1e-3)
        old_gen_scaled = old_gen_in / scale_inputs
        targets_scaled = targets_in / scale_inputs

        dist_normed = dist / scale.clamp(min=1e-3)    # [B, C_g, C_g+C_n+C_p]

        # Mask out self-distances (gen_i vs gen_i)
        diag_mask = torch.eye(C_g, dtype=torch.float32, device=gen.device)
        block_mask = F.pad(diag_mask, (0, C_n + C_p, 0, 0))  # [C_g, C_g+C_n+C_p]
        block_mask = block_mask.unsqueeze(0)                   # [1, C_g, C_g+C_n+C_p]
        dist_normed = dist_normed + block_mask * 100.0

        force_across_R = torch.zeros_like(old_gen_scaled)     # [B, C_g, S]

        split_idx = C_g + C_n

        for R in R_list:
            logits = -dist_normed / R                          # [B, C_g, T]

            affinity = F.softmax(logits, dim=-1)               # [B, C_g, T] row-softmax
            aff_transpose = F.softmax(logits, dim=-2)          # [B, C_g, T] col-softmax

            affinity = torch.sqrt((affinity * aff_transpose).clamp(min=1e-6))
            affinity = affinity * targets_w_in[:, None, :]     # weight by target importance

            aff_neg = affinity[:, :, :split_idx]               # [B, C_g, C_g+C_n]
            aff_pos = affinity[:, :, split_idx:]               # [B, C_g, C_p]

            sum_pos = aff_pos.sum(dim=-1, keepdim=True)        # [B, C_g, 1]
            sum_neg = aff_neg.sum(dim=-1, keepdim=True)        # [B, C_g, 1]

            r_coeff_neg = -aff_neg * sum_pos                   # repel negatives
            r_coeff_pos = aff_pos * sum_neg                    # attract positives

            R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)  # [B, C_g, T]

            # Force: weighted sum of target positions minus self
            total_force_R = torch.einsum("bit,bts->bis", R_coeff, targets_scaled)
            total_coeffs = R_coeff.sum(dim=-1)                 # [B, C_g]
            total_force_R = total_force_R - total_coeffs.unsqueeze(-1) * old_gen_scaled

            f_norm_val = (total_force_R ** 2).mean()
            info[f"loss_{R}"] = f_norm_val

            force_scale = f_norm_val.clamp(min=1e-8).sqrt()
            force_across_R = force_across_R + total_force_R / force_scale

        goal_scaled = old_gen_scaled + force_across_R
        return goal_scaled, scale_inputs, info

    with torch.no_grad():
        goal_scaled, scale_inputs, info = calculate_scaled_goal_and_factor(
            old_gen, targets, targets_w
        )

    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    loss = (diff ** 2).mean(dim=(-1, -2))  # [B]

    info = {k: v.mean() for k, v in info.items()}
    return loss, info
