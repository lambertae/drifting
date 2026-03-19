from __future__ import annotations

import time
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.jax_utils import replicate as R
from jax.experimental import multihost_utils
import jax.experimental.multihost_utils as mu

from utils.logging import log_for_0
from dataset.dataset import epoch0_sampler
from utils.hsdp_util import pad_and_merge, ddp_shard
from utils.env import IMAGENET_FID_NPZ, IMAGENET_PR_NPZ


INCEPTION_NET = None
_DATASET_STATS = {
    "imagenet256": IMAGENET_FID_NPZ,
}
_PR_REF_PATH = IMAGENET_PR_NPZ


def _canonical_dataset_name(name: str) -> str:
    n = name.lower()
    if "imagenet256" in n:
        return "imagenet256"
    raise ValueError(f"Only ImageNet is supported now, got: {name}")


def _build_jax_inception(batch_size=200):
    """Create the pmap-compiled Inception network used for FID/IS features."""
    # Delay these imports until after distributed init. Several upstream Flax/JAX
    # helpers construct PRNG keys at import time, which counts as a JAX
    # computation and breaks multihost initialization ordering.
    from .jax_fid import inception, resize
    from .jax_fid.cvt import load_all as load_inception_params

    model = inception.InceptionV3(pretrained=True, include_head=True, transform_input=False)
    params = R(load_inception_params())

    def apply_fn(p, x):
        return model.apply(p, x, train=False)

    fake_x = jnp.zeros((jax.local_device_count(), batch_size, 299, 299, 3), dtype=jnp.float32)
    fn = jax.pmap(apply_fn).lower(params, fake_x).compile()
    return {"params": params, "fn": fn}


def _to_local_cpu(jax_array):
    """Gather addressable shards of a JAX array into a single numpy array on CPU.

    Returns:
        np.ndarray with shape ``(local_devices * per_device_batch, ...)``.
    """
    local_shards = jax_array.addressable_shards
    local_arrays = [np.array(s.data) for s in local_shards]
    return np.concatenate(local_arrays, axis=0)


def _to_uint8(samples):
    """Convert float ``[0, 1]`` samples to ``uint8 [0, 255]``."""
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=0.0)
    return (samples * 255).clip(0, 255).astype(np.uint8)


def _revert_pmap_shape(x):
    """Flatten pmap leading dims ``(devices, batch, ...)`` to ``(devices*batch, ...)``."""
    return x.reshape((-1, *x.shape[2:]))


def _compute_stats(samples_uint8: np.ndarray, num_samples: int, *, compute_logits: bool, compute_features: bool, masks=None):
    """Run Inception over generated samples and compute dataset statistics.

    Args:
        samples_uint8: generated images as `NHWC` or `NCHW` uint8 arrays.
        num_samples: target number of valid samples after removing padding.
        compute_logits: whether to keep classifier logits for IS.
        compute_features: whether to keep raw pool features for PR.
        masks: optional validity mask with shape `(N,)`; padded samples should be `0`.
    """
    global INCEPTION_NET
    if INCEPTION_NET is None:
        INCEPTION_NET = _build_jax_inception()

    if samples_uint8.shape[-1] != 3:
        samples_uint8 = samples_uint8.transpose(0, 2, 3, 1)

    if masks is None:
        masks = np.ones((len(samples_uint8),), dtype=np.float32)

    ldc = jax.local_device_count()
    batch_size = 200
    full_batch = batch_size * ldc
    pad = int(np.ceil(len(samples_uint8) / full_batch)) * full_batch - len(samples_uint8)
    if pad > 0:
        samples_uint8 = np.concatenate([samples_uint8, np.zeros((pad, *samples_uint8.shape[1:]), dtype=np.uint8)], axis=0)
        masks = np.concatenate([masks, np.zeros(pad, dtype=masks.dtype)])

    feats_list = []
    logits_list = []
    for i in range(0, len(samples_uint8), full_batch):
        # Inception expects NHWC float input in [0, 255]; resize helper consumes BCHW.
        from .jax_fid import resize

        x = torch.from_numpy(samples_uint8[i : i + full_batch].astype(np.float32).transpose(0, 3, 1, 2))
        x = resize.forward(x).numpy().transpose(0, 2, 3, 1)
        x = x.reshape((ldc, -1, *x.shape[1:]))
        pooled, _, logits = INCEPTION_NET["fn"](INCEPTION_NET["params"], jax.lax.stop_gradient(x))
        feats_list.append(_revert_pmap_shape(pooled))
        if compute_logits and logits is not None:
            logits_list.append(_revert_pmap_shape(logits))

    feats = jnp.concatenate(feats_list)
    all_feats = multihost_utils.process_allgather(feats).reshape(-1, feats.shape[-1])
    all_feats = jax.device_get(all_feats)

    np_mask = jnp.array(masks)
    all_masks = multihost_utils.process_allgather(np_mask).reshape(-1)
    all_masks = jax.device_get(all_masks)
    valid_len = min(all_feats.shape[0], all_masks.shape[0])
    all_feats = all_feats[:valid_len]
    all_masks = all_masks[:valid_len]
    all_feats = all_feats[all_masks > 0.5][:num_samples]

    feats64 = all_feats.astype(np.float64)
    out = {
        "mu": np.mean(feats64, axis=0),
        "sigma": np.cov(feats64, rowvar=False),
    }

    if compute_features:
        out["features"] = all_feats

    if compute_logits and logits_list:
        logits = jnp.concatenate(logits_list)
        all_logits = multihost_utils.process_allgather(logits).reshape(-1, logits.shape[-1])
        all_logits = jax.device_get(all_logits)
        all_logits = all_logits[:valid_len]
        all_logits = all_logits[all_masks > 0.5][:num_samples]
        out["logits"] = all_logits

    return out


def _compute_inception_score(logits, splits=10):
    rng = np.random.RandomState(2020)
    logits = logits[rng.permutation(logits.shape[0]), :]
    probs = jax.nn.softmax(logits, axis=-1)
    probs = np.asarray(probs, dtype=np.float64)

    n = probs.shape[0]
    split_size = n // splits
    probs = probs[: split_size * splits]
    scores = []
    for i in range(splits):
        part = probs[i * split_size : (i + 1) * split_size]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    scores = np.asarray(scores, dtype=np.float64)
    return float(np.mean(scores)), float(np.std(scores))


def _load_ref_stats(dataset_name: str):
    canon = _canonical_dataset_name(dataset_name)
    path = _DATASET_STATS[canon]
    data = np.load(path)
    if "ref_mu" in data:
        return {"mu": data["ref_mu"], "sigma": data["ref_sigma"]}
    return {"mu": data["mu"], "sigma": data["sigma"]}


def evaluate_fid(
    dataset_name,
    gen_func,
    gen_params,
    eval_loader,
    logger,
    num_samples=5000,
    log_folder="fid",
    log_prefix="gen_model",
    eval_prc_recall=False,
    eval_isc=True,
    eval_fid=True,
    rng_eval=None,
):
    """Generate samples, run Inception statistics, and log release metrics.

    Args:
        dataset_name: Dataset identifier used to select reference statistics.
            Only ImageNet-256 is supported in this release.
        gen_func: Generation callable that accepts one merged eval batch plus the
            contents of ``gen_params`` and ``rng=...``. It must return samples in
            ``BCHW`` or ``BHWC`` format with values in ``[0, 1]``.
        gen_params: Keyword arguments forwarded into ``gen_func`` for every eval
            batch. This typically contains the EMA params and a fixed CFG scale.
        eval_loader: Iterable of ``(images, labels)`` batches. The labels are
            used to drive conditional generation; the image tensors are ignored.
        logger: Logger that receives scalar metrics via ``log_dict`` and a
            64-image preview grid via ``log_image``.
        num_samples: Number of valid generated samples to score after padding is
            removed across all hosts.
        log_folder: Top-level metric namespace written into the logger.
        log_prefix: Per-run metric prefix inside ``log_folder``.
        eval_prc_recall: Whether to compute precision/recall in addition to FID.
        eval_isc: Whether to compute Inception Score.
        eval_fid: Whether to compute FID.
        rng_eval: Base PRNGKey for deterministic evaluation sampling.

    Returns:
        Dict[str, float] containing the computed metrics. Keys may include
        ``fid``, ``isc_mean``, ``isc_std``, ``precision``, ``recall``, and
        ``fid_time`` depending on which evaluations are enabled.
    """
    from .jax_fid.fid import compute_frechet_distance
    from .jax_fid.precision_recall import compute_precision_recall

    if rng_eval is None:
        rng_eval = jax.random.PRNGKey(0)

    start = time.time()

    eval_iter = epoch0_sampler(eval_loader)
    all_samples = []
    all_masks = []
    cur = 0
    goal_bsz = None
    for i, batch in enumerate(eval_iter):
        if goal_bsz is None:
            goal_bsz = jax.tree.leaves(batch)[0].shape[0]
        # Pad the final batch so every host/device sees a static shape.
        batch, mask = pad_and_merge(batch, goal_bsz)
        rng_step = jax.random.fold_in(rng_eval, i)
        gen_samples = gen_func(batch, **gen_params, rng=rng_step)
        gen_samples = jax.device_put(gen_samples, ddp_shard())
        mask = jax.device_put(mask, ddp_shard())
        all_samples.append(_to_uint8(_to_local_cpu(gen_samples)))
        all_masks.append(_to_local_cpu(mask))
        cur += gen_samples.shape[0]
        if cur >= num_samples:
            break

    samples = np.concatenate(all_samples, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    stats = _compute_stats(samples, num_samples, compute_logits=eval_isc, compute_features=eval_prc_recall, masks=masks)
    ref = _load_ref_stats(dataset_name)

    metrics: Dict[str, float] = {}
    if eval_fid:
        metrics["fid"] = float(compute_frechet_distance(ref["mu"], stats["mu"], ref["sigma"], stats["sigma"]))
    if eval_isc and "logits" in stats:
        mean, std = _compute_inception_score(stats["logits"])
        metrics["isc_mean"] = mean
        metrics["isc_std"] = std
    if eval_prc_recall and "features" in stats:
        ref_images = np.load(_PR_REF_PATH)["arr_0"].astype(np.uint8)
        ref_stats = _compute_stats(ref_images, 10000, compute_logits=False, compute_features=True)
        precision, recall = compute_precision_recall(ref_stats["features"], stats["features"], k=3)
        metrics["precision"] = float(precision)
        metrics["recall"] = float(recall)

    metrics["fid_time"] = float(time.time() - start)
    logger.log_dict({f"{log_folder}/{log_prefix}_{k}": v for k, v in metrics.items()})
    logger.log_image(f"{log_folder}/{log_prefix}_viz", samples[:64])
    mu.sync_global_devices("fid evaluation finished")
    return metrics
