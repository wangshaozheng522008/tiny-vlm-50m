"""Exact RNG/structure checks and tolerant floating-point state comparison."""
import math

import numpy as np
import torch


def compare_tree(a, b, *, exact=False, path="state", atol=1e-6, rtol=1e-5):
    """Raise on mismatch; return the largest absolute tensor difference."""
    if type(a) is not type(b):
        raise AssertionError(f"{path}: state types differ")
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError(f"{path}: state keys differ")
        return max((compare_tree(a[k], b[k], exact=exact, path=f"{path}.{k}", atol=atol, rtol=rtol) for k in a), default=0.)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f"{path}: state lengths differ")
        return max((compare_tree(v, w, exact=exact, path=f"{path}[{i}]", atol=atol, rtol=rtol) for i, (v, w) in enumerate(zip(a, b))), default=0.)
    if torch.is_tensor(a):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(f"{path}: tensor metadata differs")
        if exact or path.endswith(".step") or not a.is_floating_point():
            if not torch.equal(a, b):
                raise AssertionError(f"{path}: tensor states differ")
        else:
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise AssertionError(f"{path}: nonfinite tensor state")
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=lambda msg: f"{path}: {msg}")
        return float((a.to(torch.float64) - b.to(torch.float64)).abs().max()) if a.numel() else 0.
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or not np.array_equal(a, b):
            raise AssertionError(f"{path}: NumPy states differ")
    elif a != b or isinstance(a, float) and not math.isfinite(a):
        raise AssertionError(f"{path}: state values differ")
    return 0.
