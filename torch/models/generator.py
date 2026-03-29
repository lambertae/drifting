"""PyTorch DiT (Diffusion Transformer) generator for Drift.

Implements LightningDiT-style architecture with:
  - Patch embedding
  - 2-D sincos positional encoding (frozen)
  - Label / class conditioning via LabelEmbedder
  - Transformer blocks with 6-way adaptive LayerNorm (adaLN-Zero)
  - Classifier-free guidance (CFG)
  - Final projection + unpatchify
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 2-D sincos positional embedding (matches JAX reference implementation)
# ---------------------------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Return ``[grid_size**2, embed_dim]`` sincos positional embedding."""
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2-D sincos"
    half = embed_dim // 2

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)      # 2 x [H, W]
    grid = np.stack(grid, axis=0)           # [2, H, W]
    grid = grid.reshape(2, 1, grid_size, grid_size)

    def _1d_embed(pos, dim):
        omega = np.arange(dim // 2, dtype=np.float64) / (dim // 2)
        omega = 1.0 / (10000 ** omega)
        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb_h = _1d_embed(grid[0], half)   # [H*W, half]
    emb_w = _1d_embed(grid[1], half)   # [H*W, half]
    return np.concatenate([emb_h, emb_w], axis=1).astype(np.float32)  # [H*W, embed_dim]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Image to patch sequence via a single linear (conv) projection."""

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, N, D]
        return self.proj(x).flatten(2).transpose(1, 2)


class LabelEmbedder(nn.Module):
    """Class-conditional embedding with dropout (for CFG).

    A separate "null" token (index ``num_classes``) is used for unconditional
    generation.
    """

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def token_drop(self, labels: torch.Tensor, force_drop: bool = False) -> torch.Tensor:
        """Replace labels with null class index with probability `dropout_prob`."""
        if force_drop:
            drop_ids = torch.ones_like(labels, dtype=torch.bool)
        else:
            drop_ids = torch.rand_like(labels, dtype=torch.float) < self.dropout_prob
        return torch.where(drop_ids, torch.full_like(labels, self.num_classes), labels)

    def forward(
        self, labels: torch.Tensor, train: bool = True, force_drop: bool = False
    ) -> torch.Tensor:
        if train or force_drop:
            labels = self.token_drop(labels, force_drop)
        return self.embedding_table(labels)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DiTBlock(nn.Module):
    """Single DiT block with adaLN-Zero conditioning (6 modulation parameters).

    Applies ``shift``, ``scale``, ``gate`` to both the self-attention and MLP
    sub-layers.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, bias=True
        )
        self.mlp = Mlp(hidden_size, hidden_features=int(hidden_size * mlp_ratio))

        # adaLN-Zero: produces (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        # Initialise the final linear to zero so gates start at 0
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def _modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # c: [B, D] condition
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        # Self-attention residual
        normed = self._modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP residual
        normed2 = self._modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(normed2)
        return x


class FinalLayer(nn.Module):
    """Final normalisation + projection to patch tokens."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(self.norm_final(x))


# ---------------------------------------------------------------------------
# Main generator module
# ---------------------------------------------------------------------------

class DiTGenerator(nn.Module):
    """DiT-based image generator matching the Drift JAX reference.

    The ``forward`` method accepts class labels and returns a dict
    ``{"samples": Tensor}`` with pixel values in ``[-1, 1]``.

    Args:
        input_size:  Spatial size of the *latent* feature map (e.g. 32 for
                     256-px images with a 8× VAE).  For pixel-space models use
                     the image resolution directly.
        patch_size:  Patch size for the patch-embed tokeniser.
        in_channels: Input channel count (4 for latent, 3 for pixels).
        out_channels: Output channels.  Defaults to ``in_channels``.
        hidden_size: Transformer hidden dimension.
        depth:       Number of transformer blocks.
        num_heads:   Number of attention heads.
        mlp_ratio:   MLP expansion factor.
        num_classes: Number of conditioning classes.
        cfg_dropout: Class-conditioning dropout probability (for CFG training).
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        out_channels: Optional[int] = None,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,
        cfg_dropout: float = 0.1,
        # Convenience aliases used in some configs
        image_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        if image_size is not None:
            input_size = image_size
        out_channels = out_channels or in_channels

        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes

        grid_size = input_size // patch_size
        self.num_patches = grid_size * grid_size

        # Patch embedding
        self.patch_embed = PatchEmbed(input_size, patch_size, in_channels, hidden_size)

        # Positional embedding (frozen sincos)
        pos_embed = get_2d_sincos_pos_embed(hidden_size, grid_size)
        self.register_buffer(
            "pos_embed",
            torch.from_numpy(pos_embed).unsqueeze(0),  # [1, N, D]
            persistent=False,
        )

        # Label conditioner
        self.label_embedder = LabelEmbedder(num_classes, hidden_size, cfg_dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])

        # Final projection
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_basic_init)

        # Patch embed projection
        nn.init.xavier_uniform_(self.patch_embed.proj.weight.view(self.patch_embed.proj.weight.shape[0], -1))
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)

        # Label embedding
        nn.init.normal_(self.label_embedder.embedding_table.weight, std=0.02)

        # Rescale MLP and attention output projections
        for block in self.blocks:
            # MLP fc2
            nn.init.xavier_uniform_(block.mlp.fc2.weight)
            nn.init.zeros_(block.mlp.fc2.bias)

    # ------------------------------------------------------------------
    # Unpatchify
    # ------------------------------------------------------------------

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange patch tokens back to an image tensor.

        Args:
            x: [B, N, patch_size**2 * C]

        Returns:
            [B, C, H, W]
        """
        p = self.patch_size
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)           # [B, C, h, p, w, p]
        return x.reshape(x.shape[0], c, h * p, w * p)

    # ------------------------------------------------------------------
    # Forward (single pass)
    # ------------------------------------------------------------------

    def _forward_once(self, x: torch.Tensor, c_embed: torch.Tensor) -> torch.Tensor:
        """Single forward pass through the transformer.

        Args:
            x:       [B, C, H, W] input latents / pixels
            c_embed: [B, D] class conditioning embedding

        Returns:
            [B, C, H, W] output
        """
        x = self.patch_embed(x) + self.pos_embed      # [B, N, D]
        for block in self.blocks:
            x = block(x, c_embed)
        x = self.final_layer(x, c_embed)              # [B, N, p²·C_out]
        return self.unpatchify(x)                     # [B, C_out, H, W]

    # ------------------------------------------------------------------
    # Public forward with CFG
    # ------------------------------------------------------------------

    def forward(
        self,
        c: torch.LongTensor,
        cfg_scale: float = 1.0,
        train: bool = True,
        x: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate samples conditioned on class labels `c`.

        Args:
            c:         Class labels [B].
            cfg_scale: Classifier-free guidance scale.  Set to 1 to disable.
            train:     Whether to apply label dropout (for training).
            x:         Optional explicit noise input [B, C, H, W].  If None,
                       noise is sampled internally.

        Returns:
            Dict with key ``"samples"`` -> [B, C, H, W] float tensor.
        """
        B = c.shape[0]
        device = c.device

        if x is None:
            x = torch.randn(
                B, self.in_channels, self.input_size, self.input_size,
                device=device,
            )

        if cfg_scale > 1.0:
            # Conditioned pass
            c_embed = self.label_embedder(c, train=train, force_drop=False)
            cond_out = self._forward_once(x, c_embed)

            # Unconditional pass (null class)
            c_unc = torch.full_like(c, self.num_classes)
            c_unc_embed = self.label_embedder.embedding_table(c_unc)
            unc_out = self._forward_once(x, c_unc_embed)

            output = unc_out + cfg_scale * (cond_out - unc_out)
        else:
            c_embed = self.label_embedder(c, train=train)
            output = self._forward_once(x, c_embed)

        return {"samples": output}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def dummy_input(self) -> Tuple[torch.LongTensor, float]:
        """Return a (labels, cfg_scale) tuple for a single-sample dry-run."""
        labels = torch.zeros(1, dtype=torch.long)
        return labels, 1.0

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Config-driven constructor
# ---------------------------------------------------------------------------

_PRESETS: Dict[str, dict] = {
    "DiT-S/2": dict(depth=12, hidden_size=384,  num_heads=6,  patch_size=2),
    "DiT-B/2": dict(depth=12, hidden_size=768,  num_heads=12, patch_size=2),
    "DiT-L/2": dict(depth=24, hidden_size=1024, num_heads=16, patch_size=2),
    "DiT-XL/2": dict(depth=28, hidden_size=1152, num_heads=16, patch_size=2),
}


def build_dit_generator(preset: str = "DiT-XL/2", **kwargs) -> DiTGenerator:
    """Convenience factory that merges a preset with override kwargs."""
    cfg = {**_PRESETS[preset], **kwargs}
    return DiTGenerator(**cfg)
