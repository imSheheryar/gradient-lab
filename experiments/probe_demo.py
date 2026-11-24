"""Run four representative networks and diagnose each from its gradient trajectory.

The four cases are chosen so that a snapshot at step 0 would misclassify one of
them: ReLU + LayerNorm at depth 32 looks healthy at initialisation and is dead
within 250 steps.

Run:  .venv/bin/python -m experiments.probe_demo
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

matplotlib.rcParams["axes.unicode_minus"] = False

from engine import Adam, Tensor
from engine.ops import softmax_cross_entropy
from engine.probe import classify, effective_rank
from experiments.depth_horizon import Config, MLP, make_xor

ROOT = Path(__file__).resolve().parent.parent
INK = "#12161C"
CREAM = "#F7F3EA"
CORAL = "#D9553C"
TEAL = "#2F6F73"
MUTED = "#6B6459"
GOLD = "#B8892B"

CASES = [
    ("sigmoid, depth 32", "sigmoid", "none", False, CORAL),
    ("sigmoid + residual", "sigmoid", "none", True, GOLD),
    ("relu, depth 32", "relu", "none", False, TEAL),
    ("relu + layernorm", "relu", "layernorm", False, "#5B5EA6"),
]

STEPS = 400
RECORD_EVERY = 20


def main() -> None:
    X, y = make_xor()
    rows, curves = [], {}

    for label, act, norm, res, colour in CASES:
        cfg = Config(act, "he_normal", norm, res, 32, 32, 0)
        rng = np.random.default_rng(1)
        model = MLP(cfg, rng)
        opt = Adam(model.parameters(), lr=3e-3)

        steps, ratios = [], []
        for step in range(STEPS + 1):
            idx = rng.integers(0, len(X), 128)
            loss = softmax_cross_entropy(model.forward(Tensor(X[idx])), y[idx])
            opt.zero_grad()
            loss.backward()
            if step % RECORD_EVERY == 0:
                norms = [float(np.linalg.norm(l.weight.grad)) for l in model.linears]
                steps.append(step)
                ratios.append(norms[0] / norms[-1])
            opt.step()

        accuracy = float((model.forward(Tensor(X)).data.argmax(axis=1) == y).mean())
        diagnosis = classify(steps, ratios)
        curves[label] = (steps, ratios, colour)
        rows.append((label, accuracy, diagnosis))

    lines = []
    for label, accuracy, diagnosis in rows:
        line = f"{label:<20} acc={accuracy:.3f}  {diagnosis}"
        lines.append(line)
        print(line)
    (ROOT / "artifacts" / "probe_demo.txt").write_text("\n".join(lines) + "\n")

    fig = plt.figure(figsize=(10.2, 6.2), facecolor=CREAM)
    ax = fig.add_axes([0.10, 0.14, 0.66, 0.63])
    ax.set_facecolor(CREAM)
    for label, (steps, ratios, colour) in curves.items():
        ax.semilogy(steps, ratios, color=colour, lw=1.9, label=label)
    ax.axhline(1e-6, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(8, 1.6e-6, "vanishing threshold", color=MUTED, fontsize=8)
    ax.set_xlabel("training step", color=MUTED, fontsize=9.5, labelpad=6)
    ax.set_ylabel("||grad layer 1|| / ||grad layer L||", color=MUTED, fontsize=9.5, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"1e{int(round(np.log10(v)))}" if v > 0 else ""))
    ax.yaxis.set_minor_locator(NullLocator())
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis="y", color="#E3DCCC", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED, loc="lower right",
              bbox_to_anchor=(0.99, 0.13))

    fig.text(0.055, 0.93, "FIG. 6", color=CORAL, fontsize=10, fontweight="bold")
    fig.text(0.055, 0.885, "A run can look healthy at step zero", color=INK,
             fontsize=19, fontweight="bold")
    fig.text(0.055, 0.845,
             "ReLU with LayerNorm starts at 0.16 and is dead by step 250 — "
             "a snapshot would miss it",
             color=MUTED, fontsize=8.8)
    fig.savefig(ROOT / "figures" / "probe_trajectories.png", dpi=200, facecolor=CREAM)
    print("saved figures/probe_trajectories.png")


if __name__ == "__main__":
    main()
