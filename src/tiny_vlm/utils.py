"""Small shared helpers."""

from __future__ import annotations

import random

import torch

BYTES_PER_PARAM = {"fp32": 4.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Human-readable device label, with the CUDA details that affect memory tuning."""
    if device.type != "cuda":
        return str(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    total_gb = props.total_memory / 1024**3
    # Turing (7.5) has fp16 tensor cores but no bf16, so autocast must use float16.
    return f"cuda:{index} ({props.name}, {total_gb:.1f} GB, sm_{props.major}{props.minor})"


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1024**2


def human_mb(num_bytes: float) -> str:
    return f"{num_bytes / 1024**2:.1f} MB"


def estimate_sizes(total_params: int) -> dict[str, float]:
    """Estimated weight-file size in MB, ignoring per-tensor quantisation scales."""
    return {name: total_params * bpp / 1024**2 for name, bpp in BYTES_PER_PARAM.items()}