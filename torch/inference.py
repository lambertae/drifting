"""FID-only inference entrypoint for PyTorch Drift."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional

import torch

from dataset.dataset import create_imagenet_split, get_postprocess_fn
from utils.distributed import is_main_process, setup_distributed
from utils.env import HF_ROOT
from utils.fid_util import evaluate_fid
from utils.logging import WandbLogger
from utils.misc import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_generator(model_path: str, config, device: torch.device):
    """Load a DiTGenerator from a checkpoint path."""
    from models.generator import DiTGenerator

    model = DiTGenerator(
        num_classes=int(config.dataset.num_classes),
        **dict(config.model),
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)
    # Support plain state-dict or our full checkpoint format
    if "ema_params" in ckpt:
        state = ckpt["ema_params"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[inference] Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[inference] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    model.eval()
    return model


def _make_gen_func(model, cfg_scale: float, postprocess_fn, device):
    """Return a gen_func compatible with evaluate_fid."""

    @torch.no_grad()
    def gen_func(batch, **_kwargs):
        _, labels = batch
        labels = labels.to(device)
        out = model(c=labels, cfg_scale=cfg_scale, train=False)
        samples = out["samples"]
        return postprocess_fn(samples)

    return gen_func


# ---------------------------------------------------------------------------
# Main inference / evaluation loop
# ---------------------------------------------------------------------------

def run_inference(
    config_path: str,
    model_path: str,
    num_samples: int = 50_000,
    cfg_scale: float = 4.0,
    batch_size: Optional[int] = None,
    workdir: str = "runs",
    use_wandb: bool = False,
    output_file: Optional[str] = None,
) -> dict:
    """Run FID evaluation for a trained generator.

    Args:
        config_path:  Path to the YAML training config.
        model_path:   Path to a ``.pt`` checkpoint file.
        num_samples:  Number of images to generate.
        cfg_scale:    CFG guidance scale.
        batch_size:   Override eval batch size from config.
        workdir:      Directory for logs.
        use_wandb:    Whether to log results to W&B.
        output_file:  If given, write metrics JSON to this path.

    Returns:
        Dict of metric name -> float.
    """
    setup_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(config_path)
    resolution = int(config.dataset.resolution)
    bs = batch_size or int(config.dataset.get("eval_batch_size", 64))

    postprocess_fn = get_postprocess_fn(
        use_latent=bool(config.dataset.get("use_latent", False)),
        use_cache=bool(config.dataset.get("use_cache", False)),
    )

    eval_loader, _, _ = create_imagenet_split(
        resolution=resolution,
        batch_size=bs,
        split="val",
        use_aug=False,
    )

    model = _load_generator(model_path, config, device)

    gen_func = _make_gen_func(model, cfg_scale=cfg_scale, postprocess_fn=postprocess_fn, device=device)

    logger = WandbLogger()
    logger.set_logging(
        config=config,
        use_wandb=use_wandb,
        workdir=workdir,
        project="drifting-inference",
    )

    metrics = evaluate_fid(
        dataset_name=f"imagenet{resolution}",
        gen_func=gen_func,
        gen_params={},
        eval_loader=eval_loader,
        logger=logger,
        num_samples=num_samples,
        log_folder="eval",
        eval_fid=True,
    )

    if is_main_process():
        print(f"[inference] Results: {metrics}", flush=True)
        if output_file:
            Path(output_file).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.finish()
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Drift inference / FID evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to generator checkpoint (.pt).")
    parser.add_argument("--num_samples", type=int, default=50_000, help="Samples to generate.")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG guidance scale.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override eval batch size.")
    parser.add_argument("--workdir", type=str, default="runs", help="Output / log directory.")
    parser.add_argument("--use_wandb", action="store_true", help="Log to Weights & Biases.")
    parser.add_argument("--output_file", type=str, default=None, help="Write metrics JSON here.")
    args = parser.parse_args()

    run_inference(
        config_path=args.config,
        model_path=args.ckpt,
        num_samples=args.num_samples,
        cfg_scale=args.cfg_scale,
        batch_size=args.batch_size,
        workdir=args.workdir,
        use_wandb=args.use_wandb,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
