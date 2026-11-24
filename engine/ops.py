"""Composite operations whose backward passes are worth deriving by hand.

These are the ops where a naive implementation silently produces NaN, and where
the fused form has a simpler gradient than the composition of its parts.
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


def softmax_cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Fused log-softmax + negative log-likelihood, mean over the batch.

    The naive path -- softmax, then log, then gather -- overflows once the
    logits exceed ~709 in float64 and underflows to log(0) when one class
    dominates. Shifting by the row maximum keeps every exponential in [0, 1]
    and makes the backward pass collapse to (p - y) / N.

    logits: (N, C) unnormalised scores
    targets: (N,) integer class indices
    """
    logits_data = logits.data
    targets = np.asarray(targets, dtype=np.int64)
    n = logits_data.shape[0]

    shifted = logits_data - logits_data.max(axis=1, keepdims=True)
    # log-sum-exp in the shifted frame; no overflow possible
    log_norm = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_probs = shifted - log_norm

    loss = -log_probs[np.arange(n), targets].mean()
    out = Tensor(loss, _children=(logits,), _op="softmax_ce")

    def _backward() -> None:
        probs = np.exp(log_probs)
        probs[np.arange(n), targets] -= 1.0
        logits.grad += out.grad * probs / n

    out._backward = _backward
    return out


def layernorm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    """Layer normalisation over the last axis.

    x: (N, D), gamma: (D,), beta: (D,)
    """
    mu = x.data.mean(axis=-1, keepdims=True)
    centred = x.data - mu
    var = (centred**2).mean(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = centred * inv_std

    out_data = x_hat * gamma.data + beta.data
    out = Tensor(out_data, _children=(x, gamma, beta), _op="layernorm")

    d = x.data.shape[-1]

    def _backward() -> None:
        g = out.grad
        d_xhat = g * gamma.data

        # gradient through the variance
        d_var = (d_xhat * centred * -0.5 * inv_std**3).sum(axis=-1, keepdims=True)
        # gradient through the mean (the sum of centred terms is zero, so the
        # second term of the quotient rule vanishes)
        d_mu = (-d_xhat * inv_std).sum(axis=-1, keepdims=True)

        d_x = d_xhat * inv_std + d_var * 2.0 * centred / d + d_mu / d
        x.grad += d_x
        gamma.grad += (g * x_hat).sum(axis=0)
        beta.grad += g.sum(axis=0)

    out._backward = _backward
    return out


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Plain softmax, for when you need the distribution rather than the loss."""
    shifted = x.data - x.data.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    probs = e / e.sum(axis=axis, keepdims=True)
    out = Tensor(probs, _children=(x,), _op="softmax")

    def _backward() -> None:
        # dL/dz = p * (dL/dp - sum(dL/dp * p))
        dot = (out.grad * probs).sum(axis=axis, keepdims=True)
        x.grad += probs * (out.grad - dot)

    out._backward = _backward
    return out
