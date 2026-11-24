"""Every op is checked against central finite differences at float64."""

import numpy as np
import pytest

from engine import Tensor, check
from engine.ops import layernorm, softmax, softmax_cross_entropy

rng = np.random.default_rng(0)
TOL = 1e-6  # relative error; central differences at eps=1e-6 give ~1e-9 on smooth ops


def assert_gradcheck(fn, *tensors, tol=TOL, eps=1e-6):
    for t in tensors:
        result = check(fn, t, eps=eps)
        assert result["relative_error"] < tol, (
            f"gradcheck failed for {t._op or 'leaf'}: "
            f"rel_err={result['relative_error']:.3e}"
        )


def test_add_broadcast():
    x = Tensor(rng.standard_normal((4, 3)))
    b = Tensor(rng.standard_normal(3))
    assert_gradcheck(lambda: ((x + b) ** 2).sum(), x, b)


def test_mul_broadcast():
    x = Tensor(rng.standard_normal((5, 2)))
    s = Tensor(rng.standard_normal(2) + 2.0)
    assert_gradcheck(lambda: (x * s).sum(), x, s)


def test_division():
    x = Tensor(rng.standard_normal((3, 4)) + 2.0)
    y = Tensor(rng.standard_normal((3, 4)) + 2.0)
    assert_gradcheck(lambda: (x / y).sum(), x, y)


def test_power():
    x = Tensor(np.abs(rng.standard_normal((3, 3))) + 0.5)
    assert_gradcheck(lambda: (x**3).sum(), x)


def test_matmul_chain():
    x = Tensor(rng.standard_normal((6, 4)))
    w = Tensor(rng.standard_normal((4, 5)))
    b = Tensor(rng.standard_normal(5))
    assert_gradcheck(lambda: (x @ w + b).tanh().sum(), x, w, b)


def test_batched_matmul():
    x = Tensor(rng.standard_normal((3, 4, 5)))
    w = Tensor(rng.standard_normal((5, 2)))
    assert_gradcheck(lambda: (x @ w).sum(), x, w)


def test_activations():
    x = Tensor(rng.standard_normal((4, 5)))
    for name in ("relu", "tanh", "sigmoid", "gelu", "exp"):
        # keep relu away from its kink
        xt = Tensor(np.abs(x.data) + 0.3) if name == "relu" else x
        assert_gradcheck(lambda t=xt, n=name: getattr(t, n)().sum(), xt)


def test_log():
    x = Tensor(np.abs(rng.standard_normal((3, 3))) + 0.5)
    assert_gradcheck(lambda: x.log().sum(), x)


def test_sum_axes():
    x = Tensor(rng.standard_normal((4, 3, 2)))
    assert_gradcheck(lambda: (x.sum(axis=1) ** 2).sum(), x)
    assert_gradcheck(lambda: (x.sum(axis=1, keepdims=True) ** 2).sum(), x)
    assert_gradcheck(lambda: (x.sum() ** 2), x)


def test_mean():
    x = Tensor(rng.standard_normal((4, 6)))
    assert_gradcheck(lambda: (x.mean(axis=1) ** 2).sum(), x)


def test_reshape_and_transpose():
    x = Tensor(rng.standard_normal((4, 6)))
    w = Tensor(rng.standard_normal((6, 2)))
    assert_gradcheck(lambda: ((x.reshape(2, 12) @ w.reshape(12, 1)) ** 2).sum(), x, w)
    w2 = Tensor(rng.standard_normal((4, 2)))
    assert_gradcheck(lambda: ((x.T @ w2) ** 2).sum(), x, w2)


def test_softmax_cross_entropy():
    logits = Tensor(rng.standard_normal((5, 4)) * 3.0)
    targets = np.array([0, 3, 1, 2, 0])
    assert_gradcheck(lambda: softmax_cross_entropy(logits, targets), logits)


def test_softmax_ce_extreme_logits():
    """Large logits must not produce NaN, and the gradient must stay finite."""
    logits = Tensor(np.array([[1000.0, -1000.0, 0.0], [-1000.0, 1000.0, 0.0]]))
    targets = np.array([0, 1])
    loss = softmax_cross_entropy(logits, targets)
    loss.backward()
    assert np.isfinite(loss.data)
    assert np.all(np.isfinite(logits.grad))
    assert loss.data < 1e-6  # confident and correct


def test_softmax():
    x = Tensor(rng.standard_normal((4, 5)))
    assert_gradcheck(lambda: (softmax(x) ** 2).sum(), x)


def test_layernorm():
    x = Tensor(rng.standard_normal((5, 6)))
    gamma = Tensor(rng.standard_normal(6) + 1.0)
    beta = Tensor(rng.standard_normal(6))
    assert_gradcheck(lambda: (layernorm(x, gamma, beta) ** 2).sum(), x, gamma, beta)


def test_mlp_composite():
    x = Tensor(rng.standard_normal((8, 4)))
    w1 = Tensor(rng.standard_normal((4, 7)) * 0.5)
    b1 = Tensor(rng.standard_normal(7) * 0.1)
    w2 = Tensor(rng.standard_normal((7, 3)) * 0.5)
    b2 = Tensor(rng.standard_normal(3) * 0.1)
    targets = rng.integers(0, 3, size=8)

    def loss():
        h = (x @ w1 + b1).gelu()
        logits = h @ w2 + b2
        return softmax_cross_entropy(logits, targets)

    assert_gradcheck(loss, x, w1, b1, w2, b2)
