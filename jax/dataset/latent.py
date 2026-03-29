"""Latent cache dataset and cache builder for ImageNet release workflows."""

from __future__ import annotations

import argparse
import os
import multiprocessing as mp
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "tpu,cpu")

import jax
import jax.numpy as jnp
import jax.experimental.multihost_utils as mu
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm

from utils.env import IMAGENET_CACHE_PATH, IMAGENET_PATH


@dataclass(frozen=True)
class _CacheWriteItem:
    output_path: str
    moments: np.ndarray
    moments_flip: np.ndarray


def _write_cache_file(item: _CacheWriteItem) -> None:
    output_path = Path(item.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp.{os.getpid()}")
    torch.save(
        {
            "moments": item.moments,
            "moments_flip": item.moments_flip,
        },
        tmp_path,
    )
    os.replace(tmp_path, output_path)


class LatentDataset(datasets.DatasetFolder):
    """ImageFolder-style dataset for cached latent `.pt` files."""

    def __init__(self, root: str):
        super().__init__(root=root, loader=str, extensions=(".pt",))

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        data = torch.load(path, map_location="cpu", weights_only=False)
        moments = data["moments"] if torch.rand(1) < 0.5 else data["moments_flip"]
        return np.asarray(moments), target


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """ADM-style center crop used before encoding to latent."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def _center_crop_256(image: Image.Image) -> Image.Image:
    return center_crop_arr(image, 256)


class OriginalImageFolder(datasets.ImageFolder):
    """ImageFolder that also returns class/file relative path for cache writing."""

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        rel_path = os.path.join(*path.split(os.path.sep)[-2:])
        return sample, target, rel_path


def _prepare_batch_data(images: torch.Tensor) -> np.ndarray:
    """Convert `(B,C,H,W)` torch tensor to host numpy for local-device sharding."""
    return images.numpy()


def create_cached_dataset(
    local_batch_size: int,
    target_path: str,
    data_path: str,
    *,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    save_workers: int = 0,
) -> None:
    """Encode ImageNet train/val images and write latent cache files."""
    from dataset.vae import vae_enc_decode
    from utils.hsdp_util import set_global_mesh

    local_tpu_devices = jax.local_devices(backend="tpu")
    n_local_devices = max(1, len(local_tpu_devices))

    if local_batch_size % n_local_devices != 0:
        raise ValueError(
            f"`local_batch_size` must be divisible by local TPU device count={n_local_devices}, got {local_batch_size}."
        )

    set_global_mesh(min(8, n_local_devices * jax.process_count()))
    Path(target_path, "train").mkdir(parents=True, exist_ok=True)
    Path(target_path, "val").mkdir(parents=True, exist_ok=True)
    if jax.process_count() > 1:
        mu.sync_global_devices("latent cache target dirs ready")

    # Reuse the training-style replicated TPU params path so all local devices
    # can participate in the cache build.
    encode_fn, _ = vae_enc_decode(replicate_params=True)

    local_mesh = Mesh(np.array(local_tpu_devices), axis_names=("data",))
    sample_sharding = NamedSharding(local_mesh, P("data", None, None, None))
    rng_sharding = NamedSharding(local_mesh, P("data", None))
    output_sharding = NamedSharding(local_mesh, P("data", None, None, None))
    per_device_batch = local_batch_size // n_local_devices

    @partial(
        jax.jit,
        in_shardings=(sample_sharding, rng_sharding),
        out_shardings={
            "moments": output_sharding,
            "moments_flip": output_sharding,
        },
    )
    def encode(samples, rngs):
        # Data is sharded across local devices, while the VAE params stay
        # replicated. Reshape once inside the jitted region so each device
        # processes its local microbatch with the same encode_fn used in train.
        samples = samples.reshape((n_local_devices, per_device_batch, *samples.shape[1:]))

        def _encode_shard(sample_shard, rng_shard):
            return {
                "moments": encode_fn(sample_shard, rng_shard),
                "moments_flip": encode_fn(jnp.flip(sample_shard, axis=3), rng_shard),
            }

        encoded = jax.vmap(_encode_shard, in_axes=(0, 0), out_axes=0)(samples, rngs)
        return jax.tree_util.tree_map(
            lambda x: x.reshape((local_batch_size, *x.shape[2:])),
            encoded,
        )

    transform = transforms.Compose(
        [
            transforms.Lambda(_center_crop_256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    save_pool = None
    save_futures: list[Future] = []
    if save_workers > 0:
        save_pool = ProcessPoolExecutor(max_workers=save_workers, mp_context=mp.get_context("spawn"))

    global_batch_size = local_batch_size * max(1, jax.process_count())
    process_slice_start = jax.process_index() * local_batch_size
    process_slice_end = process_slice_start + local_batch_size

    for split in ("train", "val"):
        dataset = OriginalImageFolder(os.path.join(data_path, split), transform=transform)
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": global_batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": False,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
            loader_kwargs["multiprocessing_context"] = "spawn"
        loader = torch.utils.data.DataLoader(**loader_kwargs)

        base_rng = jax.random.PRNGKey(0)
        for step, (samples, _, rel_paths) in tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"cache:{split}:host{jax.process_index()}",
        ):
            step_rng = jax.random.fold_in(base_rng, step)
            step_rng = jax.random.split(step_rng, n_local_devices)

            n_valid_global = samples.shape[0]
            rel_paths = list(rel_paths)
            if n_valid_global != global_batch_size:
                pad = global_batch_size - n_valid_global
                samples = torch.cat([samples, torch.zeros((pad,) + samples.shape[1:], dtype=samples.dtype)], dim=0)
                rel_paths.extend([""] * pad)

            local_samples = samples[process_slice_start:process_slice_end]
            encoded_local = encode(
                jax.device_put(_prepare_batch_data(local_samples), sample_sharding),
                jax.device_put(step_rng, rng_sharding),
            )
            encoded_local = jax.tree_util.tree_map(np.asarray, encoded_local)
            encoded = {
                "moments": mu.process_allgather(encoded_local["moments"], tiled=True),
                "moments_flip": mu.process_allgather(encoded_local["moments_flip"], tiled=True),
            }

            write_items = []
            for i, rel_path in enumerate(rel_paths[:n_valid_global]):
                if not rel_path:
                    continue
                output_path = str(Path(target_path, split, rel_path).with_suffix(".pt"))
                write_items.append(
                    _CacheWriteItem(
                        output_path=output_path,
                        moments=np.asarray(encoded["moments"][i]),
                        moments_flip=np.asarray(encoded["moments_flip"][i]),
                    )
                )
            if save_pool is None:
                for item in write_items:
                    _write_cache_file(item)
            else:
                save_futures.extend(save_pool.submit(_write_cache_file, item) for item in write_items)

        if jax.process_count() > 1:
            mu.sync_global_devices(f"latent cache split {split} encoded")

    if save_pool is not None:
        for future in tqdm(save_futures, desc="cache:flush", disable=jax.process_index() != 0):
            future.result()
        save_pool.shutdown()

    if jax.process_count() > 1:
        mu.sync_global_devices("latent cache files flushed")


def build_cache_from_args(args: argparse.Namespace) -> None:
    from utils.misc import run_init

    run_init()
    create_cached_dataset(
        local_batch_size=int(args.local_batch_size),
        target_path=args.target_path,
        data_path=args.data_path,
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        pin_memory=bool(args.pin_memory),
        save_workers=int(args.save_workers),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ImageNet latent cache files for release generator configs.")
    parser.add_argument("--data-path", default=IMAGENET_PATH, help="ImageNet root containing train/ and val/.")
    parser.add_argument("--target-path", default=IMAGENET_CACHE_PATH, help="Output cache root for latent .pt files.")
    parser.add_argument(
        "--local-batch-size",
        type=int,
        default=128,
        help="Per-process cache batch size. Must divide jax.local_device_count().",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker count.")
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor when num_workers > 0.",
    )
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory for the cache build.")
    parser.add_argument(
        "--save-workers",
        type=int,
        default=0,
        help="Optional process count for asynchronous latent file writes on each host.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    build_cache_from_args(parse_args(argv))


if __name__ == "__main__":
    main()
