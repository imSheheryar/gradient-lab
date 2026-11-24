"""Diagnostics for a training run.

The lesson from the depth-horizon grid is that a single measurement at step 0 is
not enough. The ReLU + LayerNorm network at depth 32 starts with a perfectly
healthy gradient ratio of 0.16 and is dead by step 250 -- so a probe that only
looks at initialisation calls a failing run healthy.

Everything here therefore takes a trajectory, not a snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Diagnosis:
    verdict: str
    start: float
    end: float
    slope: float
    detail: str

    def __str__(self) -> str:
        return (f"{self.verdict:<10} start={self.start:>8.1e}  end={self.end:>8.1e}  "
                f"slope={self.slope:>+7.4f} dec/step  {self.detail}")


def log_slope(steps, ratios) -> float:
    """Least-squares slope of log10(ratio) against step, in decades per step."""
    x = np.asarray(steps, dtype=float)
    y = np.log10(np.clip(np.asarray(ratios, dtype=float), 1e-30, 1e30))
    if len(x) < 2:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def effective_rank(activations: np.ndarray) -> float:
    """Participation ratio of the singular values: how many directions carry the signal."""
    centred = activations - activations.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centred, compute_uv=False)
    if singular.sum() == 0:
        return 0.0
    return float(singular.sum() ** 2 / (singular**2).sum())


def classify(steps, ratios) -> Diagnosis:
    """Name the failure mode from the trajectory of the first/last layer ratio."""
    ratios = np.asarray(ratios, dtype=float)
    if not np.all(np.isfinite(ratios)):
        return Diagnosis("diverged", float("nan"), float("nan"), float("nan"),
                         "non-finite gradient encountered")

    start = float(ratios[0])
    tail = ratios[len(ratios) // 2:]
    end = float(np.median(tail))
    slope = log_slope(steps, ratios)

    if start > 1e-3 and end < 1e-5:
        return Diagnosis("collapsing", start, end, slope,
                         "healthy at step 0, decays during training")
    if end < 1e-6:
        return Diagnosis("vanishing", start, end, slope,
                         "first layer receives no signal")
    if end > 1e2:
        return Diagnosis("exploding", start, end, slope,
                         "gradient grows with depth")
    if end < 1e-2:
        return Diagnosis("attenuated", start, end, slope,
                         "signal survives but is suppressed")
    return Diagnosis("healthy", start, end, slope, "gradient reaches the first layer")
