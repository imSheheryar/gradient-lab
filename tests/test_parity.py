"""The same computation, built twice, compared against torch.

This is the test that separates "my gradients look plausible" from "my
gradients are the same numbers PyTorch produces".
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from engine import SGD, Adam, AdamW, Linear, Tensor, layernorm

torch.set_default_dtype(torch.float64)
RTOL, ATOL = 1e-10, 1e-12


def test_linear_forward_and_backward_match_torch():
    rng = np.random.default_rng(0)
    x, w, b = (rng.standard_normal(s) for s in ((4, 3), (3, 5), (5,)))

    layer = Linear(3, 5, init="naive", rng=rng)
    layer.weight.data = w.copy()
    layer.bias.data = b.copy()

    x_ours = Tensor(x.copy())
    out = layer(x_ours)
    (out**2).sum().backward()

    x_t = torch.tensor(x, requires_grad=True)
    w_t = torch.tensor(w, requires_grad=True)
    b_t = torch.tensor(b, requires_grad=True)
    out_t = x_t @ w_t + b_t
    (out_t**2).sum().backward()

    np.testing.assert_allclose(out.data, out_t.detach().numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(x_ours.grad, x_t.grad.numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(layer.weight.grad, w_t.grad.numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(layer.bias.grad, b_t.grad.numpy(), rtol=RTOL, atol=ATOL)


def test_layernorm_matches_torch():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, 4))
    gamma = rng.standard_normal(4) + 1.0
    beta = rng.standard_normal(4)

    g_ours, b_ours = Tensor(gamma.copy()), Tensor(beta.copy())
    x_ours = Tensor(x.copy())
    out = layernorm(x_ours, g_ours, b_ours, eps=1e-5)
    (out**2).sum().backward()

    x_t = torch.tensor(x, requires_grad=True)
    g_t = torch.tensor(gamma, requires_grad=True)
    b_t = torch.tensor(beta, requires_grad=True)
    out_t = torch.nn.functional.layer_norm(x_t, (4,), g_t, b_t, eps=1e-5)
    (out_t**2).sum().backward()

    np.testing.assert_allclose(out.data, out_t.detach().numpy(), rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(x_ours.grad, x_t.grad.numpy(), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(g_ours.grad, g_t.grad.numpy(), rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(b_ours.grad, b_t.grad.numpy(), rtol=1e-9, atol=1e-11)


def _optimiser_parity(make_ours, make_torch, steps=5):
    rng = np.random.default_rng(2)
    w0 = rng.standard_normal((3, 4))
    scale = rng.standard_normal((3, 4))

    w_ours = Tensor(w0.copy())
    opt = make_ours([w_ours])

    w_t = torch.tensor(w0, requires_grad=True)
    opt_t = make_torch([w_t])

    for _ in range(steps):
        opt.zero_grad()
        ((w_ours * w_ours) * Tensor(scale)).sum().backward()
        opt.step()

        opt_t.zero_grad()
        ((w_t * w_t) * torch.tensor(scale)).sum().backward()
        opt_t.step()

        np.testing.assert_allclose(w_ours.data, w_t.detach().numpy(), rtol=1e-9, atol=1e-11)


def test_sgd_momentum_nesterov_matches_torch():
    _optimiser_parity(
        lambda p: SGD(p, lr=0.05, momentum=0.9, nesterov=True, weight_decay=0.01),
        lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, nesterov=True, weight_decay=0.01),
    )


def test_adam_matches_torch():
    _optimiser_parity(
        lambda p: Adam(p, lr=0.01, betas=(0.9, 0.999), eps=1e-8),
        lambda p: torch.optim.Adam(p, lr=0.01, betas=(0.9, 0.999), eps=1e-8),
    )


def test_adam_with_weight_decay_matches_torch():
    _optimiser_parity(
        lambda p: Adam(p, lr=0.01, weight_decay=0.1),
        lambda p: torch.optim.Adam(p, lr=0.01, weight_decay=0.1),
    )


def test_adamw_matches_torch():
    _optimiser_parity(
        lambda p: AdamW(p, lr=0.01, weight_decay=0.1),
        lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.1),
    )
