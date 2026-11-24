"""Run gradcheck on every op and print the error it actually measured.

Run:  .venv/bin/python -m scripts.verify
"""

from __future__ import annotations

import numpy as np

from engine import Tensor, check
from engine.ops import layernorm, softmax, softmax_cross_entropy

rng = np.random.default_rng(42)


def build_cases():
    x = Tensor(rng.standard_normal((5, 4)))
    b = Tensor(rng.standard_normal(4))
    w = Tensor(rng.standard_normal((4, 3)))
    pos = Tensor(np.abs(rng.standard_normal((4, 4))) + 0.5)
    logits = Tensor(rng.standard_normal((6, 5)) * 2.0)
    targets = rng.integers(0, 5, size=6)
    gamma = Tensor(rng.standard_normal(4) + 1.0)
    beta = Tensor(rng.standard_normal(4))
    ln_x = Tensor(rng.standard_normal((6, 4)))

    return [
        ("add (broadcast)", lambda: ((x + b) ** 2).sum(), [x, b]),
        ("multiply", lambda: (x * b).sum(), [x, b]),
        ("matmul", lambda: (x @ w).sum(), [x, w]),
        ("relu", lambda: pos.relu().sum(), [pos]),
        ("tanh", lambda: x.tanh().sum(), [x]),
        ("sigmoid", lambda: x.sigmoid().sum(), [x]),
        ("gelu", lambda: x.gelu().sum(), [x]),
        ("exp", lambda: x.exp().sum(), [x]),
        ("log", lambda: pos.log().sum(), [pos]),
        ("sum(axis=0)", lambda: (x.sum(axis=0) ** 2).sum(), [x]),
        ("mean(axis=1)", lambda: (x.mean(axis=1) ** 2).sum(), [x]),
        ("max(axis=1)", lambda: x.max(axis=1).sum(), [x]),
        ("reshape", lambda: (x.reshape(2, 10) ** 2).sum(), [x]),
        ("transpose", lambda: (x.T @ x).sum(), [x]),
        ("softmax", lambda: (softmax(x) ** 2).sum(), [x]),
        ("softmax_cross_entropy", lambda: softmax_cross_entropy(logits, targets), [logits]),
        ("layernorm", lambda: (layernorm(ln_x, gamma, beta) ** 2).sum(), [ln_x, gamma, beta]),
    ]


def main() -> None:
    print(f"{'operation':<26}{'max rel. error':>16}{'grad norm':>12}")
    print("-" * 54)
    worst = 0.0
    for name, fn, tensors in build_cases():
        errs, norms = [], []
        for t in tensors:
            result = check(fn, t)
            errs.append(result["relative_error"])
            norms.append(result["grad_norm"])
        worst = max(worst, max(errs))
        print(f"{name:<26}{max(errs):>16.2e}{max(norms):>12.2f}")
    print("-" * 54)
    print(f"worst case: {worst:.2e}   tolerance: 1.0e-06   "
          f"{'PASS' if worst < 1e-6 else 'FAIL'}")


if __name__ == "__main__":
    main()
