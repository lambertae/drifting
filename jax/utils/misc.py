from __future__ import annotations

import os
import random
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import yaml


# adapted from https://github.com/NVlabs/edm
class EasyDict(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value):
        self[name] = value


def _dict_to_easydict(d):
    if not isinstance(d, dict):
        return d
    out = EasyDict()
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _dict_to_easydict(v)
        elif isinstance(v, list):
            out[k] = [_dict_to_easydict(i) for i in v]
        else:
            out[k] = v
    return out


def load_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return _dict_to_easydict(yaml.safe_load(f))


def prepare_rng(rng_key, tags=("params", "dropout")):
    keys = jax.random.split(rng_key, len(tags))
    return dict(zip(tags, keys))


_did_run_init = False


def run_init():
    global _did_run_init
    if _did_run_init:
        return
    jax.distributed.initialize()
    _did_run_init = True


_jitted_rand = {}


def ddp_rand_func(rand_type="normal", shard="ddp"):
    from utils.hsdp_util import data_shard, ddp_shard

    sharding = ddp_shard() if shard == "ddp" else data_shard()
    key = (rand_type, shard)
    if key not in _jitted_rand:
        if rand_type == "normal":
            _jitted_rand[key] = jax.jit(jax.random.normal, out_shardings=sharding, static_argnums=(1,))
        elif rand_type == "uniform":
            _jitted_rand[key] = jax.jit(jax.random.uniform, out_shardings=sharding, static_argnums=(1,))
        else:
            raise ValueError(rand_type)
    return _jitted_rand[key]


def _profile_log(report: list[str], msg: str, *, console_print: bool) -> None:
    """Append one human-readable profiling line and optionally print it."""
    report.append(msg)
    if console_print:
        print(msg, flush=True)


def _format_metric_value(value: float, suffix: str = "") -> str:
    """Format large metric values with SI-style units for console reports."""
    for unit in ("", "K", "M", "G", "T", "P"):
        if abs(value) < 1000.0:
            return f"{value:3.2f} {unit}{suffix}".rstrip()
        value /= 1000.0
    return f"{value:.2f} E{suffix}".rstrip()


def _normalize_cost_analysis(cost_analysis: Any) -> Dict[str, float]:
    """Normalize JAX cost-analysis outputs across backend/version variants."""
    if isinstance(cost_analysis, list):
        return dict(cost_analysis[0] or {})
    return dict(cost_analysis or {})


def _extract_memory_metrics(compiled: Any) -> Dict[str, float]:
    """Extract backend memory-analysis fields when available."""
    memory_analysis = compiled.memory_analysis()
    if memory_analysis is None:
        return {
            "profile/Memory_GB": 0.0,
            "profile/Weights_MB": 0.0,
            "profile/Activations_MB": 0.0,
            "profile/Output_MB": 0.0,
        }

    temp_size = float(getattr(memory_analysis, "temp_size_in_bytes", 0.0))
    output_size = float(getattr(memory_analysis, "output_size_in_bytes", 0.0))
    arg_size = float(getattr(memory_analysis, "argument_size_in_bytes", 0.0))
    alias_size = float(getattr(memory_analysis, "alias_size_in_bytes", 0.0))
    peak_bytes = temp_size + output_size + arg_size - alias_size
    return {
        "profile/Memory_GB": peak_bytes / 1e9,
        "profile/Weights_MB": arg_size / 1e6,
        "profile/Activations_MB": temp_size / 1e6,
        "profile/Output_MB": output_size / 1e6,
    }


def profile_func(
    target_fn: Callable,
    args: tuple,
    kwargs: Optional[Dict] = None,
    name: str = "Model",
    console_print: bool = True,
    hardware_peak_bw: float = 1600.0,
    actual_run: bool = False,
    n_loops: int = 10,
    print_hlo: bool = False,
):
    """Profile a jitted JAX function and return logger-friendly scalar metrics.

    The returned dict is designed for training/eval loggers. Static metrics are
    always reported when the backend exposes them:
    - ``profile/GFLOPs``: estimated per-call FLOPs in billions.
    - ``profile/MB``: estimated memory traffic in megabytes.
    - ``profile/Intensity``: FLOPs per byte accessed.
    - ``profile/Memory_GB`` / ``profile/Weights_MB`` /
      ``profile/Activations_MB`` / ``profile/Output_MB``: memory-analysis
      breakdown when supported by the backend.

    When ``actual_run=True``, dynamic benchmark metrics are added:
    - ``profile/Time_ms``: average wall time per call.
    - ``profile/BW_GBs``: achieved bandwidth from static bytes / measured time.
    - ``profile/BW_Util``: achieved bandwidth as a percentage of
      ``hardware_peak_bw``.
    - ``profile/Achieved_TFLOPS``: achieved FLOPs throughput.
    """
    import time

    kwargs = kwargs or {}
    report: list[str] = []
    metrics: Dict[str, float] = {}

    _profile_log(report, f"[Profile] Inspecting '{name}'", console_print=console_print)
    lowered = target_fn.lower(*args, **kwargs)
    if print_hlo:
        with open("model_hlo.txt", "w", encoding="utf-8") as f:
            f.write(lowered.as_text())
    compiled = lowered.compile()
    cost = _normalize_cost_analysis(compiled.cost_analysis())
    flops = float(cost.get("flops", 0.0))
    bytes_accessed = float(cost.get("bytes accessed", 0.0))
    metrics["profile/GFLOPs"] = flops / 1e9
    metrics["profile/MB"] = bytes_accessed / 1e6
    metrics["profile/Intensity"] = flops / (bytes_accessed + 1e-9)
    metrics.update(_extract_memory_metrics(compiled))

    _profile_log(
        report,
        (
            f"[Profile] Static: FLOPs={_format_metric_value(flops, 'FLOPs')}, "
            f"traffic={_format_metric_value(bytes_accessed, 'B')}, "
            f"intensity={metrics['profile/Intensity']:.4f}"
        ),
        console_print=console_print,
    )
    if metrics["profile/Memory_GB"] > 0:
        _profile_log(
            report,
            (
                "[Profile] Memory: "
                f"peak={metrics['profile/Memory_GB']:.3f} GB, "
                f"weights={metrics['profile/Weights_MB']:.2f} MB, "
                f"activations={metrics['profile/Activations_MB']:.2f} MB, "
                f"output={metrics['profile/Output_MB']:.2f} MB"
            ),
            console_print=console_print,
        )

    if actual_run:
        _profile_log(
            report,
            f"[Profile] Benchmarking {n_loops} loop(s)",
            console_print=console_print,
        )
        _ = target_fn(*args, **kwargs)
        jax.block_until_ready(_)
        t0 = time.perf_counter()
        for _ in range(n_loops):
            out = target_fn(*args, **kwargs)
            jax.block_until_ready(out)
        dt = (time.perf_counter() - t0) / n_loops
        achieved_bw = (bytes_accessed / 1e9) / dt if dt > 0 else 0.0
        metrics["profile/Time_ms"] = dt * 1000
        metrics["profile/BW_GBs"] = achieved_bw
        metrics["profile/BW_Util"] = (achieved_bw / hardware_peak_bw) * 100 if hardware_peak_bw > 0 else 0.0
        metrics["profile/Achieved_TFLOPS"] = (metrics["profile/GFLOPs"] / dt) / 1000 if dt > 0 else 0.0
        _profile_log(
            report,
            (
                "[Profile] Runtime: "
                f"time={metrics['profile/Time_ms']:.2f} ms, "
                f"bw={metrics['profile/BW_GBs']:.2f} GB/s, "
                f"bw_util={metrics['profile/BW_Util']:.2f}%, "
                f"achieved={metrics['profile/Achieved_TFLOPS']:.2f} TFLOPS"
            ),
            console_print=console_print,
        )
    return metrics
