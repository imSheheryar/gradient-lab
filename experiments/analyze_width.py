"""Does width buy depth?

For each activation and depth, find the smallest width that reaches the accuracy
threshold. If that width does not exist inside the sweep, the configuration
cannot be rescued by width at all -- which is the interesting case.

Run:  .venv/bin/python -m experiments.analyze_width
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
INK = "#12161C"
CREAM = "#F7F3EA"
CORAL = "#D9553C"
TEAL = "#2F6F73"
MUTED = "#6B6459"
GOLD = "#B8892B"
THRESHOLD = 0.90


def main() -> None:
    rows = list(csv.DictReader((ROOT / "artifacts" / "width_sweep.csv").open()))
    stats: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        key = (r["activation"], int(r["width"]), int(r["depth"]))
        stats[key].append(float(r["accuracy"]))
    mean = {k: float(np.mean(v)) for k, v in stats.items()}

    widths = sorted({int(r["width"]) for r in rows})
    depths = sorted({int(r["depth"]) for r in rows})

    print(f"{'activation':<11}{'depth':>6}{'min width':>11}{'accuracy':>10}")
    print("-" * 38)
    minima = {}
    for act in sorted({r["activation"] for r in rows}):
        for d in depths:
            found = None
            for w in widths:
                acc = mean.get((act, w, d))
                if acc is not None and acc >= THRESHOLD:
                    found = (w, acc)
                    break
            minima[(act, d)] = found
            if found:
                print(f"{act:<11}{d:>6}{found[0]:>11}{found[1]:>10.3f}")
            else:
                print(f"{act:<11}{d:>6}{'none':>11}{'--':>10}")

    (ROOT / "artifacts" / "width_for_depth.txt").write_text(
        "\n".join(f"{a}\t{d}\t{m[0] if m else 'none'}" for (a, d), m in minima.items()) + "\n"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.6), facecolor=CREAM, sharey=True)
    for ax, act, colour in zip(axes, ("sigmoid", "relu"), (CORAL, TEAL)):
        ax.set_facecolor(CREAM)
        for d, shade in zip(depths, (0.35, 0.6, 1.0)):
            xs = [w for w in widths if (act, w, d) in mean]
            ys = [mean[(act, w, d)] for w in xs]
            ax.plot(xs, ys, marker="o", ms=4, lw=1.8, color=colour, alpha=shade,
                    label=f"depth {d}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(widths)
        ax.set_xticklabels(widths, color=MUTED, fontsize=8.5)
        ax.axhline(THRESHOLD, color=MUTED, lw=0.8, ls=(0, (4, 3)))
        ax.set_title(act, color=INK, fontsize=12, fontweight="bold", loc="left", pad=10)
        ax.set_xlabel("width", color=MUTED, fontsize=9, labelpad=6)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.grid(axis="y", color="#E3DCCC", lw=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0.45, 1.02)
    axes[0].set_ylabel("training accuracy", color=MUTED, fontsize=9, labelpad=6)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=MUTED,
                   loc="center left", bbox_to_anchor=(0.02, 0.42))
    axes[0].annotate("depth 16 and 32 never leave chance,\nat any width up to 128",
                     xy=(24, 0.505), xytext=(26, 0.60), color=MUTED, fontsize=8.2,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    axes[1].annotate("width 8 is enough at every depth",
                     xy=(16, 0.995), xytext=(20, 0.74), color=MUTED, fontsize=8.2,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

    fig.text(0.045, 0.93, "FIG. 4", color=CORAL, fontsize=10, fontweight="bold")
    fig.text(0.045, 0.885, "Width is not a substitute for depth", color=INK,
             fontsize=18, fontweight="bold")
    fig.text(0.045, 0.845,
             "same task, same 400 steps, He init, no normalisation: only the width changes",
             color=MUTED, fontsize=8.8)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.78, bottom=0.12, wspace=0.08)
    fig.savefig(ROOT / "figures" / "width_vs_depth.png", dpi=200, facecolor=CREAM)
    print("saved figures/width_vs_depth.png")


if __name__ == "__main__":
    main()
