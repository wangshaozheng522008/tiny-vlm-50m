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


def human_mb(num_bytes: float) -> str:
    return f"{num_bytes / 1024**2:.1f} MB"


def estimate_sizes(total_params: int) -> dict[str, float]:
    """Estimated weight-file size in MB, ignoring per-tensor quantisation scales."""
    return {name: total_params * bpp / 1024**2 for name, bpp in BYTES_PER_PARAM.items()}