"""PyTorch distributed utilities (replaces JAX hsdp_util)."""
from __future__ import annotations
import os
import torch
import torch.distributed as dist


def setup_distributed() -> None:
    """Initialize process group from environment variables (torchrun sets these)."""
    if dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def log_for_0(msg: str, *args) -> None:
    if is_main_process():
        if args:
            print(msg % args, flush=True)
        else:
            print(msg, flush=True)
