"""PyTorch MAE with ResNet backbone for the Drift feature extractor.

Implements a Masked Autoencoder (MAE) that:
  - Encodes an image via a ResNet backbone (multi-scale features)
  - Masks a random subset of patches
  - Decodes masked patches with a shallow transformer decoder
  - Optionally includes a classification head (cross-entropy loss)
"""
from __future__ import annotations
import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ---------------------------------------------------------------------------
# Activation extraction hook utility
# ---------------------------------------------------------------------------

class _HookActivations:
    """Context-manager / callable that captures named layer outputs via hooks."""

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self._handles = []
        self.activations: Dict[str, torch.Tensor] = {}
        for name, module in model.named_modules():
            if name in layer_names:
                handle = module.register_forward_hook(self._make_hook(name))
                self._handles.append(handle)

    def _make_hook(self, name: str) -> Callable:
        def hook(module, input, output):
            self.activations[name] = output
        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __del__(self):
        self.remove()


def build_activation_function(
    model: nn.Module, layer_names: Optional[List[str]] = None
) -> Callable[[torch.Tensor], Dict[str, torch.Tensor]]:
    """Return a callable that runs `model` and returns activations from `layer_names`.

    Args:
        model:       PyTorch module (e.g. a ResNet backbone).
        layer_names: Names of sub-modules whose outputs to capture.  If None,
                     a sensible default set of ResNet stages is used.

    Returns:
        ``extract(x) -> {layer_name: tensor}``
    """
    if layer_names is None:
        # Default: capture all four ResNet stages
        layer_names = ["layer1", "layer2", "layer3", "layer4"]

    hooks = _HookActivations(model, layer_names)

    def extract(x: torch.Tensor) -> Dict[str, torch.Tensor]:
        hooks.activations.clear()
        model(x)
        return {k: v for k, v in hooks.activations.items()}

    # Attach hook reference to prevent GC
    extract._hooks = hooks  # type: ignore[attr-defined]
    return extract


# ---------------------------------------------------------------------------
# Positional embedding
# ---------------------------------------------------------------------------

def _sincos_1d(length: int, dim: int) -> torch.Tensor:
    omega = torch.arange(dim // 2, dtype=torch.float32) / (dim // 2)
    omega = 1.0 / (10000 ** omega)                          # [D/2]
    pos = torch.arange(length, dtype=torch.float32)         # [L]
    out = torch.einsum("i,d->id", pos, omega)               # [L, D/2]
    return torch.cat([out.sin(), out.cos()], dim=-1)        # [L, D]


def _sincos_2d(h: int, w: int, dim: int) -> torch.Tensor:
    assert dim % 4 == 0
    half = dim // 2
    emb_h = _sincos_1d(h, half)   # [H, D/2]
    emb_w = _sincos_1d(w, half)   # [W, D/2]
    # Broadcast and concat
    emb = torch.cat([
        emb_h.unsqueeze(1).expand(-1, w, -1),
        emb_w.unsqueeze(0).expand(h, -1, -1),
    ], dim=-1)                    # [H, W, D]
    return emb.reshape(h * w, dim)


# ---------------------------------------------------------------------------
# MAE Decoder (lightweight transformer)
# ---------------------------------------------------------------------------

class MAEDecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + a
        return x + self.mlp(self.norm2(x))


class MAEDecoder(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        patch_size: int,
        out_channels: int,
        num_patches: int,
        depth: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.proj = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_dim), requires_grad=False
        )
        self.blocks = nn.ModuleList([
            MAEDecoderBlock(decoder_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_size * patch_size * out_channels)

    def _init_pos_embed(self, h: int, w: int) -> None:
        embed = _sincos_2d(h, w, self.pos_embed.shape[-1])
        self.pos_embed.data.copy_(embed.unsqueeze(0))

    def forward(
        self,
        encoded: torch.Tensor,     # [B, num_visible, encoder_dim]
        visible_ids: torch.Tensor, # [B, num_visible] long
        num_patches: int,
    ) -> torch.Tensor:
        B, _, D = self.proj(encoded).shape
        projected = self.proj(encoded)                      # [B, V, decoder_dim]

        # Build full token sequence with mask tokens inserted
        full_tokens = self.mask_token.expand(B, num_patches, -1).clone()
        full_tokens.scatter_(
            1,
            visible_ids.unsqueeze(-1).expand(-1, -1, full_tokens.shape[-1]),
            projected,
        )
        full_tokens = full_tokens + self.pos_embed

        for block in self.blocks:
            full_tokens = block(full_tokens)
        full_tokens = self.norm(full_tokens)
        return self.pred(full_tokens)                       # [B, num_patches, p²·C]


# ---------------------------------------------------------------------------
# Main MAE model
# ---------------------------------------------------------------------------

class MAEResNetPyTorch(nn.Module):
    """Masked Autoencoder with a ResNet backbone encoder.

    The encoder is a ResNet-50 (by default) whose ``layer4`` feature map is
    flattened into patch tokens.  A shallow transformer decoder reconstructs
    the masked tokens in pixel space.

    Args:
        num_classes:     Number of ImageNet classes (for optional cls head).
        image_size:      Input image spatial resolution.
        patch_size:      Spatial patch size (applied on the feature map grid).
        encoder_backbone: ``"resnet50"`` or ``"resnet34"`` etc.
        decoder_dim:     Hidden size of the MAE decoder.
        decoder_depth:   Number of decoder transformer blocks.
        decoder_heads:   Number of decoder attention heads.
        in_channels:     Input image channels.
        norm_pix_loss:   Whether to normalise target pixels per patch.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        image_size: int = 256,
        patch_size: int = 16,
        encoder_backbone: str = "resnet50",
        decoder_dim: int = 512,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        in_channels: int = 3,
        norm_pix_loss: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.norm_pix_loss = norm_pix_loss
        self.num_classes = num_classes

        self.num_patches_h = image_size // patch_size
        self.num_patches_w = image_size // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # ---------------------------------------------------------------- encoder
        backbone_fn = getattr(tvm, encoder_backbone, None)
        if backbone_fn is None:
            raise ValueError(f"Unknown backbone: {encoder_backbone}")
        backbone = backbone_fn(weights=None)

        # Remove the average pool and fc head; keep everything up to layer4
        self.encoder_stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.encoder_layer1 = backbone.layer1
        self.encoder_layer2 = backbone.layer2
        self.encoder_layer3 = backbone.layer3
        self.encoder_layer4 = backbone.layer4

        # Determine encoder output channels from a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            feat = self._encode_backbone(dummy)
        encoder_dim = feat.shape[-1]
        self.encoder_dim = encoder_dim

        # Positional embedding for encoder tokens
        self.enc_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, encoder_dim), requires_grad=False
        )

        # Projection from feature spatial size to patch grid (adaptive pool)
        self.enc_pool = nn.AdaptiveAvgPool2d((self.num_patches_h, self.num_patches_w))

        # ---------------------------------------------------------------- decoder
        self.decoder = MAEDecoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            patch_size=patch_size,
            out_channels=in_channels,
            num_patches=self.num_patches,
            depth=decoder_depth,
            num_heads=decoder_heads,
        )
        self.decoder._init_pos_embed(self.num_patches_h, self.num_patches_w)

        # Optional classification head
        self.cls_head = nn.Linear(encoder_dim, num_classes) if num_classes > 0 else None

        self._init_weights()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        # Sincos positional embeddings
        enc_pos = _sincos_2d(self.num_patches_h, self.num_patches_w, self.encoder_dim)
        self.enc_pos_embed.data.copy_(enc_pos.unsqueeze(0))

        if self.cls_head is not None:
            nn.init.normal_(self.cls_head.weight, std=0.01)
            nn.init.zeros_(self.cls_head.bias)

    # ------------------------------------------------------------------
    # Backbone encoder
    # ------------------------------------------------------------------

    def _encode_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone and return pooled patch tokens [B, N, C]."""
        x = self.encoder_stem(x)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)                    # [B, C, h, w]
        x = self.enc_pool(x)                          # [B, C, pH, pW]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)              # [B, H*W, C]
        return x

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def _random_mask(
        self,
        tokens: torch.Tensor,
        mask_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly mask `mask_ratio` fraction of tokens.

        Returns:
            visible_tokens:  [B, num_visible, C]
            visible_ids:     [B, num_visible] long indices
            mask:            [B, N] bool (True = masked)
        """
        B, N, C = tokens.shape
        num_masked = int(mask_ratio * N)
        num_visible = N - num_masked

        noise = torch.rand(B, N, device=tokens.device)
        ids_shuffle = noise.argsort(dim=1)              # ascending: visible first
        ids_visible = ids_shuffle[:, :num_visible]      # [B, num_visible]

        visible_tokens = tokens.gather(
            1, ids_visible.unsqueeze(-1).expand(-1, -1, C)
        )

        mask = torch.ones(B, N, dtype=torch.bool, device=tokens.device)
        mask.scatter_(1, ids_visible, False)            # False = visible

        return visible_tokens, ids_visible, mask

    # ------------------------------------------------------------------
    # Patch target
    # ------------------------------------------------------------------

    def _patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Extract patch pixel values from images.

        Args:
            imgs: [B, C, H, W]

        Returns:
            [B, N, patch_size² * C]
        """
        p = self.patch_size
        B, C, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 1, 3, 5)           # [B, h, w, C, p, p]
        return x.reshape(B, h * w, C * p * p)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        train: bool = True,
        mask_ratio_min: float = 0.5,
        mask_ratio_max: float = 0.75,
        lambda_cls: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute MAE (+ optional cls) loss.

        Args:
            x:               Input images [B, C, H, W] in ``[-1, 1]``.
            labels:          Class labels [B].
            train:           Whether in training mode.
            mask_ratio_min:  Minimum mask ratio (uniform sample).
            mask_ratio_max:  Maximum mask ratio (uniform sample).
            lambda_cls:      Weight for the classification cross-entropy loss.

        Returns:
            (loss, metrics)  where ``loss`` is a scalar Tensor and
            ``metrics`` is a dict of scalar Tensors.
        """
        B = x.shape[0]

        # Encode
        tokens = self._encode_backbone(x) + self.enc_pos_embed  # [B, N, C]

        # Sample mask ratio
        if train:
            ratio = mask_ratio_min + torch.rand(1).item() * (mask_ratio_max - mask_ratio_min)
        else:
            ratio = (mask_ratio_min + mask_ratio_max) / 2.0

        visible_tokens, visible_ids, mask = self._random_mask(tokens, mask_ratio=ratio)

        # Decode
        pred = self.decoder(visible_tokens, visible_ids, self.num_patches)  # [B, N, p²C]

        # MAE reconstruction loss (only on masked patches)
        target = self._patchify(x)   # [B, N, p²C]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        mae_loss = ((pred - target) ** 2).mean(dim=-1)  # [B, N]
        mae_loss = (mae_loss * mask.float()).sum() / mask.float().sum().clamp(min=1)

        metrics: Dict[str, torch.Tensor] = {"mae_loss": mae_loss.detach()}
        total_loss = mae_loss

        # Optional classification loss
        if lambda_cls > 0.0 and self.cls_head is not None:
            # Global average pool of encoder tokens
            cls_feat = tokens.mean(dim=1)               # [B, C]
            logits = self.cls_head(cls_feat)             # [B, num_classes]
            cls_loss = F.cross_entropy(logits, labels)
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            metrics["cls_loss"] = cls_loss.detach()
            metrics["cls_acc"] = acc.detach()
            total_loss = total_loss + lambda_cls * cls_loss

        return total_loss, metrics

    # ------------------------------------------------------------------
    # Feature extraction (for Drift)
    # ------------------------------------------------------------------

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return multi-scale backbone features and patch tokens.

        Useful for building the feature bank in Drift training.

        Args:
            x: [B, C, H, W]

        Returns:
            Dict with keys ``"tokens"`` (patch tokens) and ``"layer{1..4}"``
            (intermediate backbone activations).
        """
        feats: Dict[str, torch.Tensor] = {}

        h = self.encoder_stem(x)
        h = self.encoder_layer1(h); feats["layer1"] = h
        h = self.encoder_layer2(h); feats["layer2"] = h
        h = self.encoder_layer3(h); feats["layer3"] = h
        h = self.encoder_layer4(h); feats["layer4"] = h

        tokens = self.enc_pool(h).flatten(2).transpose(1, 2)  # [B, N, C]
        feats["tokens"] = tokens
        return feats

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def dummy_input(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (images, labels) for a single-sample dry-run."""
        x = torch.zeros(1, self.in_channels, self.image_size, self.image_size)
        labels = torch.zeros(1, dtype=torch.long)
        return x, labels

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
