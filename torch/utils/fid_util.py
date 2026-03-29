"""FID evaluation utilities for PyTorch."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import numpy as np
import torch
from tqdm import tqdm
from utils.distributed import is_main_process, get_world_size, get_rank, barrier


# ---------------------------------------------------------------------------
# Internal FID computation
# ---------------------------------------------------------------------------

def _compute_activation_statistics(images: np.ndarray, model, batch_size: int = 50, device: str = "cpu"):
    """Compute Inception activation statistics (mu, sigma) over `images` (NHWC uint8)."""
    model.eval()
    all_acts = []
    n = images.shape[0]
    for start in range(0, n, batch_size):
        batch = images[start: start + batch_size]
        # NHWC uint8 -> NCHW float [0,1]
        x = torch.from_numpy(batch).permute(0, 3, 1, 2).float().div(255.0).to(device)
        with torch.no_grad():
            act = model(x)
        if isinstance(act, (list, tuple)):
            act = act[0]
        all_acts.append(act.cpu().numpy())
    acts = np.concatenate(all_acts, axis=0)
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """Numpy FID computation."""
    from scipy import linalg
    diff = mu1 - mu2
    # Stable sqrt of product of covariances
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def _load_inception(device: str = "cpu"):
    """Load Inception-v3 feature extractor (requires torchvision)."""
    try:
        from torchmetrics.image.fid import NoTrainInceptionV3
        model = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"])
        model = model.to(device)
        return model
    except ImportError:
        pass

    # Fallback: torchvision InceptionV3 with pool3 hook
    import torchvision.models as tvm
    model = tvm.inception_v3(pretrained=False, transform_input=False)
    model.fc = torch.nn.Identity()
    model = model.eval().to(device)
    return model


def compute_fid_from_stats(
    generated_images: np.ndarray,
    ref_stats_npz: str,
    device: str = "cpu",
    batch_size: int = 50,
) -> float:
    """Compute FID between `generated_images` and pre-computed reference stats.

    Args:
        generated_images: uint8 NHWC array.
        ref_stats_npz:    path to npz containing ``mu`` and ``sigma``.
        device:           device string.
        batch_size:       batch size for Inception forward passes.

    Returns:
        FID scalar.
    """
    ref = np.load(ref_stats_npz)
    mu_ref, sigma_ref = ref["mu"], ref["sigma"]

    inception = _load_inception(device)
    mu_gen, sigma_gen = _compute_activation_statistics(
        generated_images, inception, batch_size=batch_size, device=device
    )
    return _frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)


# ---------------------------------------------------------------------------
# Public evaluation entry-point
# ---------------------------------------------------------------------------

def evaluate_fid(
    dataset_name: str,
    gen_func: Callable,
    gen_params: Dict[str, Any],
    eval_loader,
    logger,
    num_samples: int = 50_000,
    log_folder: str = "fid_eval",
    log_prefix: str = "",
    eval_prc_recall: bool = False,
    eval_isc: bool = False,
    eval_fid: bool = True,
    rng_eval=None,
) -> Dict[str, float]:
    """Evaluate generative quality metrics.

    Args:
        dataset_name:   Informational string (e.g. "imagenet256").
        gen_func:       Callable ``(batch, **gen_params) -> Tensor [B,C,H,W] or [B,H,W,C]``.
        gen_params:     Keyword arguments forwarded to `gen_func`.
        eval_loader:    DataLoader providing evaluation batches.
        logger:         WandbLogger / NullLogger instance.
        num_samples:    Number of samples to generate.
        log_folder:     Prefix for logged metric keys.
        log_prefix:     Additional sub-prefix.
        eval_prc_recall: Evaluate precision & recall (not implemented; placeholder).
        eval_isc:        Evaluate Inception Score (not implemented; placeholder).
        eval_fid:        Evaluate FID.
        rng_eval:        Unused; kept for API compatibility.

    Returns:
        Dict of metric name -> float value.
    """
    from utils.env import IMAGENET_FID_NPZ

    all_samples = []
    n_collected = 0

    for batch in tqdm(eval_loader, desc="Generating samples", disable=not is_main_process()):
        if n_collected >= num_samples:
            break
        with torch.no_grad():
            samples = gen_func(batch, **gen_params)
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        all_samples.append(samples)
        n_collected += samples.shape[0]

    if not all_samples:
        return {}

    all_samples = np.concatenate(all_samples, axis=0)[:num_samples]

    # Ensure NHWC uint8
    if all_samples.ndim == 4 and all_samples.shape[1] in (1, 3):
        all_samples = np.transpose(all_samples, (0, 2, 3, 1))
    if all_samples.dtype != np.uint8:
        all_samples = np.clip(all_samples * 255.0, 0, 255).astype(np.uint8)

    metrics: Dict[str, float] = {}
    prefix = f"{log_prefix}/" if log_prefix else ""

    if eval_fid and Path(IMAGENET_FID_NPZ).exists():
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            fid_val = compute_fid_from_stats(all_samples, IMAGENET_FID_NPZ, device=device)
            metrics[f"{prefix}fid"] = fid_val
        except Exception as exc:
            metrics[f"{prefix}fid"] = -1.0
            print(f"[fid_util] FID computation failed: {exc}", flush=True)

    if metrics and is_main_process():
        logger.log_dict({f"{log_folder}/{k}": v for k, v in metrics.items()})

    return metrics
