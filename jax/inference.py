"""FID-only inference entrypoint.

Usage:
    python inference.py --init-from "hf://latent_L_sota" --workdir runs/fid
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import jax

from dataset.dataset import create_imagenet_split, get_postprocess_fn
from utils.env import HF_ROOT
from utils.fid_util import evaluate_fid
from utils.hsdp_util import data_shard, ddp_shard, set_global_mesh
from utils.init_util import load_generator_model_and_params
from utils.logging import WandbLogger
from utils.misc import prepare_rng, run_init

run_init()


def _is_latent(metadata: dict) -> bool:
    """Determine if the model operates in latent space from its metadata."""
    model_cfg = metadata.get("model_config", {})
    return model_cfg.get("in_channels", 3) == 4


def _load_model(init_from: str):
    """Build generator, load params, return (jitted_gen_step, params, metadata)."""
    model, params, metadata = load_generator_model_and_params(
        init_from,
        hf_cache_dir=HF_ROOT,
    )
    latent = _is_latent(metadata)
    postprocess_fn = get_postprocess_fn(use_aug=False, use_latent=False, use_cache=latent)
    gen_step_jit = jax.jit(partial(generate_step, apply_fn=model.apply, postprocess_fn=postprocess_fn))
    return gen_step_jit, params, metadata


def generate_step(batch, params, rng, apply_fn, postprocess_fn, cfg_scale=1.0):
    """Generate samples from a batch of labels for evaluation/demo code paths."""
    _, labels = batch
    labels = jax.lax.with_sharding_constraint(labels, data_shard())
    latent_samples = apply_fn(
        {"params": params},
        train=False,
        rngs=prepare_rng(rng, ["noise"]),
        c=labels,
        cfg_scale=cfg_scale,
    )["samples"]
    latent_samples = jax.tree_util.tree_map(
        lambda x: jax.lax.with_sharding_constraint(x, ddp_shard()),
        latent_samples,
    )
    return postprocess_fn(latent_samples)

# ---------------------------------------------------------------------------
# eval_fid
# ---------------------------------------------------------------------------

def run_eval_fid(
    gen_step_jit, params, metadata, init_from: str, workdir: str,
    *, num_samples: int, cfg_scale: float, eval_batch_size: int,
    use_wandb: bool, wandb_entity: str | None, wandb_project: str, wandb_name: str | None,
) -> dict:
    eval_loader, _, _ = create_imagenet_split(
        resolution=256, split="val",
        batch_size=eval_batch_size // jax.process_count(),
        num_workers=0,
    )

    work_path = Path(workdir).resolve()
    logger = WandbLogger()
    logger.set_logging(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_name or f"{Path(init_from).name}_fid",
        use_wandb=use_wandb,
        workdir=str(work_path),
        log_every_k=1,
    )

    metrics = evaluate_fid(
        dataset_name="imagenet256",
        gen_func=gen_step_jit,
        gen_params={"params": params, "cfg_scale": cfg_scale},
        eval_loader=eval_loader,
        logger=logger,
        num_samples=num_samples,
        log_folder="fid_eval",
        log_prefix=f"cfg_{cfg_scale:g}",
        eval_prc_recall=(num_samples >= 50000),
        eval_isc=True,
        eval_fid=True,
        rng_eval=jax.random.PRNGKey(0),
    )
    logger.finish()
    return {"init_from": init_from, "cfg_scale": cfg_scale, "metadata": metadata, **metrics}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference: FID evaluation.")
    parser.add_argument("--init-from", required=True,
                        help="hf://<name> or local checkpoint path.")
    parser.add_argument("--workdir", default="runs/infer", help="Output directory.")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="Classifier-free guidance scale.")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--hsdp-dim", type=int, default=None)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="release-fid")
    parser.add_argument("--wandb-name", type=str, default=None)
    return parser


def run_inference_from_args(args: argparse.Namespace) -> dict:
    hsdp = args.hsdp_dim or min(8, jax.local_device_count() * jax.process_count())
    set_global_mesh(hsdp)
    gen_step_jit, params, metadata = _load_model(args.init_from)
    result = run_eval_fid(
        gen_step_jit, params, metadata, args.init_from, args.workdir,
        num_samples=args.num_samples,
        cfg_scale=args.cfg_scale,
        eval_batch_size=args.eval_batch_size,
        use_wandb=args.use_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )
    return result


def main() -> None:
    args = build_parser().parse_args()
    result = run_inference_from_args(args)
    print(json.dumps(result, indent=2))
    if args.json_out:
        out = Path(args.json_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
