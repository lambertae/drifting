"""ImageNet dataset pipeline for PyTorch Drift."""
from __future__ import annotations
import os
import random
from typing import Callable, Iterator, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils.env import IMAGENET_PATH, IMAGENET_CACHE_PATH
from utils.distributed import get_rank, get_world_size


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """Center-crop `pil_image` to ``image_size x image_size``.

    Progressively halves the image until the short side is < 2*image_size, then
    up-scales with bicubic and crops.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------

def _build_transforms(resolution: int, use_aug: bool, split: str) -> transforms.Compose:
    if use_aug and split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(resolution, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    return transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img.convert("RGB"), resolution)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def _build_imagenet_dataset(
    *,
    resolution: int,
    use_aug: bool,
    use_cache: bool,
    split: str,
):
    if use_cache:
        from dataset.latent import LatentDataset  # type: ignore[import]
        return LatentDataset(root=os.path.join(IMAGENET_CACHE_PATH, split))

    transform = _build_transforms(resolution, use_aug=use_aug, split=split)
    return ImageFolder(root=os.path.join(IMAGENET_PATH, split), transform=transform)


# ---------------------------------------------------------------------------
# Worker init
# ---------------------------------------------------------------------------

def worker_init_fn(worker_id: int, rank: int = 0) -> None:
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_imagenet_split(
    *,
    resolution: int,
    batch_size: int,
    split: str,
    use_aug: bool = False,
    use_latent: bool = False,
    use_cache: bool = False,
    num_workers: int = 4,
    **kwargs,
) -> Tuple[DataLoader, Callable, Callable]:
    """Create an ImageNet DataLoader for `split` (``"train"`` or ``"val"``).

    Returns:
        (loader, preprocess_fn, postprocess_fn)
    """
    dataset = _build_imagenet_dataset(
        resolution=resolution,
        use_aug=use_aug,
        use_cache=use_cache,
        split=split,
    )

    rank = get_rank()
    world_size = get_world_size()

    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=(split == "train"),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=lambda wid: worker_init_fn(wid, rank),
            drop_last=(split == "train"),
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )

    def preprocess_fn(batch):
        images, labels = batch
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        return {"images": images, "labels": labels}

    def postprocess_fn(samples):
        """Convert float tensor in ``[-1, 1]`` to uint8 NHWC numpy array."""
        if isinstance(samples, torch.Tensor):
            x = samples.float()
            x = (x * 0.5 + 0.5).clamp(0, 1)
            x = (x * 255).to(torch.uint8)
            if x.ndim == 4 and x.shape[1] in (1, 3):
                x = x.permute(0, 2, 3, 1)
            return x.cpu().numpy()
        return samples

    return loader, preprocess_fn, postprocess_fn


def infinite_sampler(loader: DataLoader, start_step: int = 0) -> Iterator:
    """Yield batches from `loader` indefinitely, re-shuffling each epoch."""
    epoch = 0
    it = iter(loader)
    step = 0
    while True:
        try:
            batch = next(it)
        except StopIteration:
            epoch += 1
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)
            it = iter(loader)
            batch = next(it)
        if step >= start_step:
            yield batch
        step += 1


def epoch0_sampler(loader: DataLoader) -> Iterator:
    """One-epoch iterator (for evaluation)."""
    return iter(loader)


def get_postprocess_fn(
    use_aug: bool = False,
    use_latent: bool = False,
    use_cache: bool = False,
) -> Callable:
    """Return a postprocessing function converting model output to uint8 images."""

    def postprocess_fn(samples):
        if isinstance(samples, torch.Tensor):
            x = samples.float()
            if use_latent or use_cache:
                return x
            x = (x * 0.5 + 0.5).clamp(0, 1)
            x = (x * 255).to(torch.uint8)
            if x.ndim == 4 and x.shape[1] in (1, 3):
                x = x.permute(0, 2, 3, 1)
            return x.cpu().numpy()
        return samples

    return postprocess_fn
