"""Compare every layer and optimiser against PyTorch and print the deviation.

Run:  .venv/bin/python -m scripts.verify_parity
"""

from __future__ import annotations

import numpy as np
import torch

from engine import SGD, Adam, AdamW, Linear, Tensor, layernorm

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)


def report(name: str, ours: np.ndarray, theirs) -> float:
    diff = float(np.max(np.abs(ours - np.asarray(theirs))))
    print(f"{name:<34}{diff:>12.2e}")
    return diff


def main() -> None:
    print(f"{'component':<34}{'max abs diff':>12}")
    print("-" * 46)
    worst = 0.0

    # ---- Linear: forward and all three gradients
    x, w, b = (rng.standard_normal(s) for s in ((4, 3), (3, 5), (5,)))
    layer = Linear(3, 5, init="naive", rng=rng)
    layer.weight.data, layer.bias.data = w.copy(), b.copy()
    x_ours = Tensor(x.copy())
    out = layer(x_ours)
    (out**2).sum().backward()

    x_t = torch.tensor(x, requires_grad=True)
    w_t = torch.tensor(w, requires_grad=True)
    b_t = torch.tensor(b, requires_grad=True)
    out_t = x_t @ w_t + b_t
    (out_t**2).sum().backward()

    worst = max(worst, report("linear forward", out.data, out_t.detach().numpy()))
    worst = max(worst, report("linear grad wrt input", x_ours.grad, x_t.grad.numpy()))
    worst = max(worst, report("linear grad wrt weight", layer.weight.grad, w_t.grad.numpy()))
    worst = max(worst, report("linear grad wrt bias", layer.bias.grad, b_t.grad.numpy()))

    # ---- LayerNorm
    x2 = rng.standard_normal((6, 4))
    g, bb = rng.standard_normal(4) + 1.0, rng.standard_normal(4)
    g_ours, b_ours = Tensor(g.copy()), Tensor(bb.copy())
    x2_ours = Tensor(x2.copy())
    ln = layernorm(x2_ours, g_ours, b_ours, eps=1e-5)
    (ln**2).sum().backward()

    x2_t = torch.tensor(x2, requires_grad=True)
    g_t = torch.tensor(g, requires_grad=True)
    b_t2 = torch.tensor(bb, requires_grad=True)
    ln_t = torch.nn.functional.layer_norm(x2_t, (4,), g_t, b_t2, eps=1e-5)
    (ln_t**2).sum().backward()

    worst = max(worst, report("layernorm forward", ln.data, ln_t.detach().numpy()))
    worst = max(worst, report("layernorm grad wrt input", x2_ours.grad, x2_t.grad.numpy()))
    worst = max(worst, report("layernorm grad wrt gamma", g_ours.grad, g_t.grad.numpy()))

    # ---- Optimisers, five steps on the same quadratic
    def run_ours(make_opt) -> np.ndarray:
        p = Tensor(w0.copy())
        opt = make_opt([p])
        for _ in range(5):
            opt.zero_grad()
            ((p * p) * Tensor(scale)).sum().backward()
            opt.step()
        return p.data.copy()

    def run_torch(make_opt) -> np.ndarray:
        p = torch.tensor(w0, requires_grad=True)
        opt = make_opt([p])
        for _ in range(5):
            opt.zero_grad()
            ((p * p) * torch.tensor(scale)).sum().backward()
            opt.step()
        return p.detach().numpy()

    w0 = rng.standard_normal((3, 4))
    scale = rng.standard_normal((3, 4))

    worst = max(worst, report("SGD + Nesterov, 5 steps",
                              run_ours(lambda p: SGD(p, lr=0.05, momentum=0.9, nesterov=True)),
                              run_torch(lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, nesterov=True))))
    worst = max(worst, report("Adam, 5 steps",
                              run_ours(lambda p: Adam(p, lr=0.01)),
                              run_torch(lambda p: torch.optim.Adam(p, lr=0.01))))
    worst = max(worst, report("AdamW, 5 steps",
                              run_ours(lambda p: AdamW(p, lr=0.01, weight_decay=0.1)),
                              run_torch(lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.1))))

    print("-" * 46)
    print(f"worst deviation: {worst:.2e}   tolerance 1e-12   "
          f"{'PASS' if worst < 1e-12 else 'FAIL'}")


if __name__ == "__main__":
    main()
