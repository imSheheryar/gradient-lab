"""Finite-difference gradient checking.

The tolerance everyone quotes is 1e-7 relative error. That number is a claim
about float64 arithmetic on a handful of parameters; it is not achievable in
float32, and it degrades as the parameter count grows because each forward
pass carries its own rounding error. `check()` therefore reports the error it
actually measured instead of asserting a constant, so the precision study in
`experiments/precision.py` can read the same numbers.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .tensor import Tensor


def numerical_grad(
    fn: Callable[[], Tensor],
    wrt: Tensor,
    eps: float = 1e-6,
) -> np.ndarray:
    """Central-difference estimate of d(fn)/d(wrt).

    `fn` must rebuild the graph on every call and return a scalar Tensor.
    Cost is two forward passes per element of `wrt`, so this is only ever run
    on small tensors or on a random subset of a large one.
    """
    grad = np.zeros_like(wrt.data)
    it = np.nditer(wrt.data, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        original = wrt.data[idx]

        wrt.data[idx] = original + eps
        plus = float(fn().data)

        wrt.data[idx] = original - eps
        minus = float(fn().data)

        wrt.data[idx] = original
        grad[idx] = (plus - minus) / (2.0 * eps)
    return grad


def relative_error(analytical: np.ndarray, numerical: np.ndarray) -> float:
    """Normalised difference, the standard gradcheck metric."""
    denominator = np.linalg.norm(analytical) + np.linalg.norm(numerical)
    if denominator == 0.0:
        return 0.0
    return float(np.linalg.norm(analytical - numerical) / denominator)


def check(
    fn: Callable[[], Tensor],
    wrt: Tensor,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Run the analytical and numerical gradients and compare them."""
    wrt.zero_grad()
    fn().backward()
    analytical = wrt.grad.copy()
    numerical = numerical_grad(fn, wrt, eps=eps)
    return {
        "relative_error": relative_error(analytical, numerical),
        "max_abs_diff": float(np.max(np.abs(analytical - numerical))),
        "grad_norm": float(np.linalg.norm(analytical)),
    }
