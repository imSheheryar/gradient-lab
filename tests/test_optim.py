"""Optimiser behaviour that does not need PyTorch to verify."""

import numpy as np
import pytest

from engine import SGD, Adam, AdamW, Tensor


def quadratic(params):
    return sum((p * p).sum() for p in params)


def test_sgd_descends_quadratic():
    x = Tensor(np.array([3.0, -4.0]))
    opt = SGD([x], lr=0.1)
    for _ in range(100):
        opt.zero_grad()
        quadratic([x]).backward()
        opt.step()
    assert np.allclose(x.data, 0.0, atol=1e-3)


def test_momentum_is_faster_on_ill_conditioned_problem():
    def run(use_momentum: bool) -> float:
        x = Tensor(np.array([1.0, 1.0]))
        opt = SGD([x], lr=0.01, momentum=0.9 if use_momentum else 0.0)
        for _ in range(60):
            opt.zero_grad()
            # curvature 100 in the first coordinate, 1 in the second
            loss = (x * x * Tensor(np.array([100.0, 1.0]))).sum()
            loss.backward()
            opt.step()
        return float(loss.data)

    assert run(True) < run(False)


def test_adam_first_step_is_bias_corrected():
    """With a constant gradient, the first Adam step has magnitude ~lr."""
    x = Tensor(np.array([1.0]))
    opt = Adam([x], lr=0.1)
    opt.zero_grad()
    quadratic([x]).backward()  # grad = 1
    opt.step()
    assert x.data[0] == pytest.approx(0.9, abs=1e-6)


def test_adamw_decays_even_without_gradient():
    x = Tensor(np.array([1.0]))
    opt = AdamW([x], lr=0.1, weight_decay=0.5)
    opt.zero_grad()  # grad stays zero
    opt.step()
    assert x.data[0] < 1.0


def test_coupled_weight_decay_shrinks_with_zero_grad():
    x = Tensor(np.array([1.0]))
    opt = SGD([x], lr=0.1, weight_decay=1.0)
    opt.zero_grad()
    opt.step()
    assert x.data[0] == pytest.approx(0.9)


def test_nesterov_requires_momentum():
    with pytest.raises(ValueError):
        SGD([Tensor(np.ones(2))], nesterov=True)


def test_zero_grad_clears_between_steps():
    x = Tensor(np.array([2.0]))
    opt = SGD([x], lr=0.5)
    for _ in range(2):
        opt.zero_grad()
        quadratic([x]).backward()
        opt.step()
    # 2 -> 0 -> 0, not 2 -> 0 -> -2
    assert x.data[0] == pytest.approx(0.0)
