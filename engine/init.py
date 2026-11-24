"""Weight initialisation schemes.

These are variables in the depth-horizon experiment, not decoration: the whole
claim is that activation, initialisation and normalisation interact, so each
scheme has to be implemented exactly as published rather than approximated.
"""

from __future__ import annotations

import numpy as np


def naive(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """The default you get if you do not think about it: N(0, 1)."""
    return rng.standard_normal(shape)


def xavier_uniform(shape, rng, gain: float = 1.0) -> np.ndarray:
    """Glorot & Bengio (2010): limit = gain * sqrt(6 / (fan_in + fan_out))."""
    fan_in, fan_out = _fans(shape)
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape)


def xavier_normal(shape, rng, gain: float = 1.0) -> np.ndarray:
    fan_in, fan_out = _fans(shape)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return rng.standard_normal(shape) * std


def he_normal(shape, rng, gain: float = np.sqrt(2.0)) -> np.ndarray:
    """He et al. (2015): std = sqrt(2 / fan_in) for ReLU networks."""
    fan_in, _ = _fans(shape)
    return rng.standard_normal(shape) * (gain / np.sqrt(fan_in))


def he_uniform(shape, rng, gain: float = np.sqrt(2.0)) -> np.ndarray:
    fan_in, _ = _fans(shape)
    limit = gain * np.sqrt(3.0 / fan_in)
    return rng.uniform(-limit, limit, size=shape)


def zeros(shape, rng) -> np.ndarray:
    return np.zeros(shape)


def _fans(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 1:
        return shape[0], shape[0]
    fan_in, fan_out = shape[0], shape[-1]
    receptive = int(np.prod(shape[1:-1])) if len(shape) > 2 else 1
    return fan_in * receptive, fan_out * receptive


SCHEMES = {
    "naive": naive,
    "xavier_uniform": xavier_uniform,
    "xavier_normal": xavier_normal,
    "he_normal": he_normal,
    "he_uniform": he_uniform,
}


def get(name: str):
    if name not in SCHEMES:
        raise KeyError(f"unknown init scheme {name!r}; choose from {sorted(SCHEMES)}")
    return SCHEMES[name]
