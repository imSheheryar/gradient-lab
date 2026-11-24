"""Tests for the things that are easy to get subtly wrong in an autodiff engine."""

import numpy as np
import pytest

from engine import Tensor


def test_fanout_accumulates():
    """A tensor used twice must receive the sum of both gradients."""
    x = Tensor(3.0)
    y = x * x + x  # dy/dx = 2x + 1 = 7
    y.backward()
    assert x.grad == pytest.approx(7.0)


def test_diamond_graph():
    x = Tensor(2.0)
    a = x * 2.0
    b = x * 3.0
    c = a * b  # 6x^2 -> 12x = 24
    c.backward()
    assert x.grad == pytest.approx(24.0)


def test_deep_chain_does_not_recurse():
    """600 layers would blow a recursive topological sort."""
    x = Tensor(1.0)
    y = x
    for _ in range(600):
        y = y * 1.001
    y.backward()
    assert x.grad == pytest.approx(1.001**600, rel=1e-12)


def test_gradients_are_not_shared_between_calls():
    x = Tensor(np.array([1.0, 2.0]))
    (x * 2).sum().backward()
    first = x.grad.copy()
    x.zero_grad()
    (x * 2).sum().backward()
    assert np.allclose(first, x.grad)


def test_broadcast_gradient_shape():
    """The reduced gradient must match the broadcast operand, not the output."""
    x = Tensor(np.ones((4, 3)))
    b = Tensor(np.arange(3.0))
    (x + b).sum().backward()
    assert b.grad.shape == (3,)
    assert np.allclose(b.grad, 4.0)


def test_matmul_broadcast_gradient_shape():
    x = Tensor(np.ones((3, 4, 5)))
    w = Tensor(np.ones((5, 2)))
    (x @ w).sum().backward()
    assert w.grad.shape == (5, 2)
    assert np.allclose(w.grad, 3.0 * 4.0)


def test_scalar_backward_requires_gradient_for_non_scalar():
    x = Tensor(np.ones((2, 2)))
    with pytest.raises(ValueError):
        (x * 2).backward()


def test_explicit_gradient_seed():
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    y = x * 2.0
    y.backward(np.array([1.0, 0.0, 0.0]))
    assert np.allclose(x.grad, [2.0, 0.0, 0.0])


def test_sum_backward_broadcasts():
    x = Tensor(np.arange(6.0).reshape(2, 3))
    s = x.sum(axis=0)
    s.backward(np.array([1.0, 10.0, 100.0]))
    assert np.allclose(x.grad, [[1.0, 10.0, 100.0], [1.0, 10.0, 100.0]])


def test_max_routes_gradient_to_argmax():
    x = Tensor(np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 9.0]]))
    m = x.max(axis=1)
    m.backward(np.array([1.0, 1.0]))
    assert np.allclose(x.grad, [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def test_zero_grad():
    x = Tensor(np.ones(3))
    (x * 3).sum().backward()
    assert np.allclose(x.grad, 3.0)
    x.zero_grad()
    assert np.allclose(x.grad, 0.0)


def test_ties_in_max_split_gradient():
    x = Tensor(np.array([2.0, 2.0]))
    x.max().backward()
    assert np.allclose(x.grad, [0.5, 0.5])
