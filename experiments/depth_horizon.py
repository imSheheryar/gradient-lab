"""The depth-horizon grid.

For every combination of activation, initialisation, normalisation, residual
connection and depth, train an otherwise identical MLP on the same task for the
same number of steps, and record:

  accuracy      final training accuracy
  loss          final cross-entropy
  grad_ratio    ||grad of first layer|| / ||grad of last layer|| after step 0
  diverged      whether the run produced a non-finite loss

The depth horizon of a configuration is the deepest network that still reaches
the accuracy threshold. Everything else is held fixed on purpose: the point is
to attribute the difference to the intervention, not to the model size.

Run:  .venv/bin/python -m experiments.depth_horizon --steps 400
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from engine import Adam, Linear, Tensor, layernorm
from engine.ops import softmax_cross_entropy

ROOT = Path(__file__).resolve().parent.parent
ACTIVATIONS = ("sigmoid", "tanh", "relu", "gelu")
INITS = ("naive", "xavier_normal", "he_normal")
NORMS = ("none", "layernorm")
RESIDUALS = (False, True)


@dataclass(frozen=True)
class Config:
    activation: str
    init: str
    norm: str
    residual: bool
    depth: int
    width: int
    seed: int


def make_xor(n_per_quadrant: int = 400, noise: float = 0.35, seed: int = 0):
    """Four clusters, labels by sign agreement.

    Deliberately easy: a two-layer network solves it in a few hundred steps, so
    a failure at depth 32 is attributable to depth and not to the task.
    """
    rng = np.random.default_rng(seed)
    centres = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
    labels = np.array([0, 1, 1, 0])
    X = np.concatenate([c + noise * rng.standard_normal((n_per_quadrant, 2))
                        for c in centres])
    y = np.repeat(labels, n_per_quadrant)
    order = rng.permutation(len(X))
    return X[order], y[order]


def make_spirals(n_per_class: int = 750, noise: float = 0.12, seed: int = 0):
    """Two interleaved spirals: much harder, kept for the appendix."""
    rng = np.random.default_rng(seed)
    n = n_per_class
    t = np.linspace(0.25, 4.0 * np.pi, n)
    r = t / (4.0 * np.pi)
    x0 = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    x1 = np.stack([-r * np.cos(t), -r * np.sin(t)], axis=1)
    X = np.concatenate([x0, x1]) + noise * rng.standard_normal((2 * n, 2))
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    order = rng.permutation(len(X))
    return X[order], y[order]


TASKS = {"xor": make_xor, "spirals": make_spirals}


class MLP:
    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        dims = [2] + [cfg.width] * cfg.depth + [2]
        self.linears = [
            Linear(dims[i], dims[i + 1], init=cfg.init, rng=rng)
            for i in range(len(dims) - 1)
        ]
        self.gammas = [Tensor(np.ones(cfg.width)) for _ in range(cfg.depth)]
        self.betas = [Tensor(np.zeros(cfg.width)) for _ in range(cfg.depth)]

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for lin in self.linears:
            params.extend(lin.parameters())
        if self.cfg.norm == "layernorm":
            params.extend(self.gammas)
            params.extend(self.betas)
        return params

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for i, lin in enumerate(self.linears[:-1]):
            z = lin(h)
            if self.cfg.norm == "layernorm":
                z = layernorm(z, self.gammas[i], self.betas[i])
            a = getattr(z, self.cfg.activation)()
            h = h + a if (self.cfg.residual and i > 0) else a
        return self.linears[-1](h)


def run_one(cfg: Config, X: np.ndarray, y: np.ndarray, steps: int, batch: int,
            lr: float) -> dict:
    rng = np.random.default_rng(cfg.seed * 100_003 + cfg.depth)
    model = MLP(cfg, rng)
    params = model.parameters()
    opt = Adam(params, lr=lr)

    x_all = Tensor(X)
    grad_ratio = float("nan")
    loss_value = float("nan")
    diverged = False
    start = time.time()

    for step in range(steps):
        idx = rng.integers(0, len(X), batch)
        loss = softmax_cross_entropy(model.forward(Tensor(X[idx])), y[idx])
        opt.zero_grad()
        loss.backward()

        if step == 0:
            first = np.linalg.norm(model.linears[0].weight.grad)
            last = np.linalg.norm(model.linears[-1].weight.grad)
            grad_ratio = float(first / last) if last > 0 else float("inf")

        opt.step()
        loss_value = float(loss.data)
        if not np.isfinite(loss_value):
            diverged = True
            break

    if diverged:
        accuracy = float(max(y.mean(), 1 - y.mean()))
    else:
        logits = model.forward(x_all)
        accuracy = float((logits.data.argmax(axis=1) == y).mean())

    return {
        **asdict(cfg),
        "accuracy": accuracy,
        "loss": loss_value if np.isfinite(loss_value) else float("nan"),
        "grad_ratio": grad_ratio,
        "diverged": int(diverged),
        "seconds": time.time() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--widths", type=int, nargs="+", default=[32])
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--activations", nargs="+", default=list(ACTIVATIONS))
    parser.add_argument("--inits", nargs="+", default=list(INITS))
    parser.add_argument("--norms", nargs="+", default=list(NORMS))
    parser.add_argument("--residuals", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--task", choices=sorted(TASKS), default="xor")
    parser.add_argument("--out", default="artifacts/depth_horizon.csv")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple] = set()
    if out_path.exists():
        with out_path.open() as fh:
            for row in csv.DictReader(fh):
                done.add((row["activation"], row["init"], row["norm"],
                          row["residual"], int(row["width"]),
                          int(row["depth"]), int(row["seed"])))

    grid = list(itertools.product(args.activations, args.inits, args.norms,
                                  [bool(r) for r in args.residuals],
                                  args.widths, args.depths, args.seeds))
    todo = [g for g in grid
            if (g[0], g[1], g[2], str(g[3]), g[4], g[5], g[6]) not in done]
    if args.limit:
        todo = todo[: args.limit]

    X, y = TASKS[args.task]()
    print(f"task: {args.task}, {len(X)} points | grid {len(grid)} runs | "
          f"{len(todo)} to do | steps {args.steps}", flush=True)

    write_header = not out_path.exists()
    start = time.time()
    with out_path.open("a", newline="") as fh:
        writer = None
        for i, (act, init, norm, residual, width, depth, seed) in enumerate(todo, 1):
            cfg = Config(act, init, norm, residual, depth, width, seed)
            row = run_one(cfg, X, y, args.steps, args.batch, args.lr)
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
            writer.writerow(row)
            fh.flush()
            if i % 10 == 0 or i == len(todo):
                elapsed = time.time() - start
                eta = elapsed / i * (len(todo) - i)
                print(f"[{i:>4}/{len(todo)}] {act:<8}{init:<15}{norm:<10}"
                      f"res={int(residual)} w={width:<4} d={depth:<3} seed={seed}  "
                      f"acc={row['accuracy']:.3f}  {elapsed:.0f}s elapsed, "
                      f"{eta:.0f}s left", flush=True)

    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
