"""Revalidate persisted quantized tensors against their FP32 source."""

import torch

from .model import MiniLLM
from .quantize import quantize_model


def verify_quantized_tensors(payload: dict, source: MiniLLM, *, bits: int = 4,
                             group_size: int = 128) -> None:
    """Recompute the CPU export and authenticate all tensor values and metadata."""
    expected = quantize_model(source.cpu(), bits, group_size)

    def check(actual, reference, key):
        if isinstance(reference, dict):
            if not isinstance(actual, dict) or actual.keys() != reference.keys():
                raise ValueError(f"quantized source mismatch at {key}: keys")
            for name, value in reference.items():
                check(actual[name], value, f"{key}.{name}")
        elif isinstance(reference, torch.Tensor):
            if (not isinstance(actual, torch.Tensor) or actual.dtype != reference.dtype
                    or actual.shape != reference.shape or not torch.equal(actual, reference)):
                raise ValueError(f"quantized source mismatch at {key}: tensor")
        elif type(actual) is not type(reference) or actual != reference:
            raise ValueError(f"quantized source mismatch at {key}: metadata")

    check({key: payload.get(key) for key in expected}, expected, "INT4")
