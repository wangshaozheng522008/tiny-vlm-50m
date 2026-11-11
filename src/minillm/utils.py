"""Shared helpers: seeding, device selection, size formatting, memory probes."""

from __future__ import annotations

import random

import numpy as np
import torch

BYTES_PER_PARAM = {"fp32": 4.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return str(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return f"cuda:{index} ({props.name}, {props.total_memory / 1024**3:.1f} GB, sm_{props.major}{props.minor})"


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1024**2


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (Linux reports kilobytes)."""
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def human_bytes(num_bytes: float) -> str:
    """Decimal MB and binary MiB side by side, because the 50 MB budget is decimal."""
    return f"{num_bytes / 1_000_000:.2f} MB ({num_bytes / 1024**2:.2f} MiB)"


def estimate_size_bytes(total_params: int, bytes_per_param: float) -> int:
    return int(total_params * bytes_per_param)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
