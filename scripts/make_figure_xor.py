"""Figure 0: the engine solving XOR, drawn as a decision surface.

Run:  .venv/bin/python -m scripts.make_figure_xor
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from engine import Tensor
from engine.ops import softmax_cross_entropy

INK = "#12161C"
CREAM = "#F7F3EA"
CORAL = "#D9553C"
TEAL = "#2F6F73"
MUTED = "#6B6459"


def main() -> None:
    rng = np.random.default_rng(0)

    n_per = 120
    centres = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
    labels = np.array([0, 1, 1, 0])
    X = np.concatenate([c + 0.35 * rng.standard_normal((n_per, 2)) for c in centres])
    y = np.repeat(labels, n_per)

    x = Tensor(X)
    w1 = Tensor(rng.standard_normal((2, 16)) * 0.8)
    b1 = Tensor(np.zeros(16))
    w2 = Tensor(rng.standard_normal((16, 16)) * 0.8)
    b2 = Tensor(np.zeros(16))
    w3 = Tensor(rng.standard_normal((16, 2)) * 0.8)
    b3 = Tensor(np.zeros(2))
    params = [w1, b1, w2, b2, w3, b3]

    lr = 0.15
    for _ in range(601):
        h1 = (x @ w1 + b1).tanh()
        h2 = (h1 @ w2 + b2).tanh()
        logits = h2 @ w3 + b3
        loss = softmax_cross_entropy(logits, y)
        for p in params:
            p.zero_grad()
        loss.backward()
        for p in params:
            p.data -= lr * p.grad

    # decision surface on a grid
    g = np.linspace(-2.2, 2.2, 260)
    gx, gy = np.meshgrid(g, g)
    grid = Tensor(np.stack([gx.ravel(), gy.ravel()], axis=1))
    h1 = (grid @ w1 + b1).tanh()
    h2 = (h1 @ w2 + b2).tanh()
    logits = h2 @ w3 + b3
    prob = np.exp(logits.data - logits.data.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    prob = prob[:, 0].reshape(gx.shape)

    train_logits = ((x @ w1 + b1).tanh() @ w2 + b2).tanh() @ w3 + b3
    acc = float((train_logits.data.argmax(axis=1) == y).mean())

    fig, ax = plt.subplots(figsize=(7.2, 7.2), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    surface = LinearSegmentedColormap.from_list("ink", [TEAL, "#E8E2D3", CORAL])
    ax.contourf(gx, gy, prob, levels=np.linspace(0, 1, 13), cmap=surface, alpha=0.5)
    ax.contour(gx, gy, prob, levels=[0.5], colors=[INK], linewidths=1.4)

    for label, colour, name in ((0, TEAL, "class 0"), (1, CORAL, "class 1")):
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], s=13, color=colour, edgecolor=CREAM,
                   linewidth=0.4, alpha=0.95, label=name, zorder=3)

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=MUTED)

    fig.text(0.06, 0.945, "FIG. 0", color=CORAL, fontsize=10, fontweight="bold")
    fig.text(0.06, 0.905, "XOR, solved by an engine with no framework under it",
             color=INK, fontsize=16, fontweight="bold")
    fig.text(0.06, 0.868,
             f"2-16-16-2 tanh MLP, hand-derived backward pass, 600 SGD steps   |   "
             f"accuracy {acc:.3f}   |   loss {float(loss.data):.4f}",
             color=MUTED, fontsize=8.8)

    fig.subplots_adjust(left=0.06, right=0.96, top=0.83, bottom=0.05)
    out = "figures/xor_decision_boundary.png"
    fig.savefig(out, dpi=200, facecolor=CREAM)
    print(f"saved {out}  accuracy={acc:.3f}")


if __name__ == "__main__":
    main()
