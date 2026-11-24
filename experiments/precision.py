"""Where does the 1e-7 gradcheck tolerance come from, and when does it fail?

Finite differences have two competing error sources: truncation, which falls as
eps grows, and floating-point cancellation, which explodes as eps shrinks. The
sweet spot depends on the dtype, so the tolerance everyone quotes is really a
statement about float64 arithmetic on a small number of parameters.

Run:  .venv/bin/python -m experiments.precision
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

matplotlib.rcParams["axes.unicode_minus"] = False

from engine import Tensor, relative_error
from engine.ops import softmax_cross_entropy

ROOT = Path(__file__).resolve().parent.parent
INK = "#12161C"
CREAM = "#F7F3EA"
CORAL = "#D9553C"
TEAL = "#2F6F73"
MUTED = "#6B6459"

EPSILONS = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5,
            3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8]


def build(dtype):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, 4)).astype(dtype)
    W1 = (rng.standard_normal((4, 8)) * 0.5).astype(dtype)
    W2 = (rng.standard_normal((8, 3)) * 0.5).astype(dtype)
    y = rng.integers(0, 3, size=16)

    tW1 = Tensor(W1.copy())
    tb1 = Tensor(np.zeros(8, dtype=dtype))
    tW2 = Tensor(W2.copy())
    tb2 = Tensor(np.zeros(3, dtype=dtype))
    tx = Tensor(X)

    def loss() -> Tensor:
        hidden = (tx @ tW1 + tb1).tanh()
        return softmax_cross_entropy(hidden @ tW2 + tb2, y)

    return loss, tW1


def analytical(loss, param) -> np.ndarray:
    param.zero_grad()
    loss().backward()
    return param.grad.copy()


def numerical(loss, param, eps: float) -> np.ndarray:
    grad = np.zeros_like(param.data)
    flat = param.data.reshape(-1)
    out = grad.reshape(-1)
    for i in range(flat.size):
        original = flat[i]
        flat[i] = original + eps
        plus = float(loss().data)
        flat[i] = original - eps
        minus = float(loss().data)
        flat[i] = original
        out[i] = (plus - minus) / (2.0 * eps)
    return grad


def main() -> None:
    results = {}
    truth = None

    for dtype in (np.float64, np.float32):
        loss, param = build(dtype)
        grad = analytical(loss, param)
        if truth is None:
            truth = grad.astype(np.float64)
        curve = []
        for eps in EPSILONS:
            num = numerical(loss, param, eps)
            err = relative_error(grad.astype(np.float64), num.astype(np.float64))
            curve.append(err)
        results[dtype] = (curve, grad)
        best = min(curve)
        best_eps = EPSILONS[curve.index(best)]
        print(f"{np.dtype(dtype).name:<8} analytical vs float64 truth: "
              f"{relative_error(grad.astype(np.float64), truth):.2e}   "
              f"best finite-difference error: {best:.2e} at eps={best_eps:g}")

    fig = plt.figure(figsize=(9.8, 6.0), facecolor=CREAM)
    ax = fig.add_axes([0.105, 0.15, 0.66, 0.62])
    ax.set_facecolor(CREAM)

    for dtype, colour, label in ((np.float64, TEAL, "float64"),
                                 (np.float32, CORAL, "float32")):
        curve = results[dtype][0]
        ax.loglog(EPSILONS, curve, marker="o", ms=4, lw=1.8, color=colour, label=label)
        best = min(curve)
        best_eps = EPSILONS[curve.index(best)]
        offset = (best_eps * 2.4, best * 2.0) if dtype is np.float64 else (best_eps * 2.4, best * 0.55)
        ax.annotate(f"{best:.0e}", xy=(best_eps, best), xytext=offset,
                    color=colour, fontsize=9, fontweight="bold")

    ax.annotate("perturbation lost to\nfloat32 rounding", xy=(2e-7, 1.0), xytext=(1.1e-7, 3.2e-1),
                color=CORAL, fontsize=8.2,
                arrowprops=dict(arrowstyle="-", color=CORAL, lw=0.8))
    ax.axhline(1e-7, color=MUTED, lw=1.0, ls=(0, (5, 3)))
    ax.text(1.6e-1, 1.6e-7, "the 1e-7 tolerance the brief asserts", color=MUTED, fontsize=8.5)
    ax.set_xlabel("finite-difference step  eps", color=MUTED, fontsize=9.5, labelpad=6)
    ax.set_ylabel("relative gradient error", color=MUTED, fontsize=9.5, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    exp_fmt = FuncFormatter(lambda v, _: f"1e{int(round(np.log10(v)))}" if v > 0 else "")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, numticks=12))
        axis.set_major_formatter(exp_fmt)
        axis.set_minor_locator(NullLocator())
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(color="#E3DCCC", lw=0.7, which="major")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="lower left")

    fig.text(0.055, 0.93, "FIG. 5", color=CORAL, fontsize=10, fontweight="bold")
    fig.text(0.055, 0.885, "1e-7 is a float64 claim", color=INK, fontsize=19, fontweight="bold")
    fig.text(0.055, 0.845,
             "gradient check on the same 4-8-3 network, same weights, only the dtype changes",
             color=MUTED, fontsize=8.8)
    fig.savefig(ROOT / "figures" / "precision.png", dpi=200, facecolor=CREAM)
    print("saved figures/precision.png")

    with (ROOT / "artifacts" / "precision.txt").open("w") as fh:
        fh.write("eps\tfloat64\tfloat32\n")
        for i, eps in enumerate(EPSILONS):
            fh.write(f"{eps:g}\t{results[np.float64][0][i]:.6e}\t"
                     f"{results[np.float32][0][i]:.6e}\n")


if __name__ == "__main__":
    main()
