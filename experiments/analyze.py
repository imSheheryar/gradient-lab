"""Aggregate the depth-horizon grid into a table and two figures.

Run:  .venv/bin/python -m experiments.analyze
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


def load(path: Path) -> list[dict]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["depth"] = int(r["depth"])
        r["seed"] = int(r["seed"])
        r["width"] = int(r["width"])
        r["residual"] = r["residual"] == "True"
        r["accuracy"] = float(r["accuracy"])
        r["grad_ratio"] = float(r["grad_ratio"])
    return rows


def aggregate(rows: list[dict], keys: tuple[str, ...]):
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = {}
    for key, items in groups.items():
        accs = np.array([i["accuracy"] for i in items])
        ratios = np.array([i["grad_ratio"] for i in items])
        out[key] = {
            "accuracy": float(accs.mean()),
            "sd": float(accs.std()),
            "grad_ratio": float(np.nanmedian(ratios)),
            "n": len(items),
        }
    return out


def horizon(stats, activation, init, norm, residual, depths) -> int:
    best = 0
    for d in sorted(depths):
        key = (activation, init, norm, residual, d)
        if key in stats and stats[key]["accuracy"] >= THRESHOLD:
            best = d
    return best


def main() -> None:
    rows = load(ROOT / "artifacts" / "depth_horizon.csv")
    depths = sorted({r["depth"] for r in rows})
    stats = aggregate(rows, ("activation", "init", "norm", "residual", "depth"))

    # ---------------------------------------------------------------- horizon table
    lines = []
    header = f"{'activation':<11}{'init':<16}{'norm':<11}{'res':<6}{'horizon':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for act in ("sigmoid", "tanh", "relu", "gelu"):
        for init in ("naive", "xavier_normal", "he_normal"):
            for norm in ("none", "layernorm"):
                for res in (False, True):
                    h = horizon(stats, act, init, norm, res, depths)
                    lines.append(f"{act:<11}{init:<16}{norm:<11}{int(res):<6}{h:>8}")
    table = "\n".join(lines)
    (ROOT / "artifacts" / "depth_horizon_table.txt").write_text(table + "\n")
    print(table)

    # ---------------------------------------------------------------- heatmap
    configs = []
    for act in ("sigmoid", "tanh", "relu", "gelu"):
        for norm in ("none", "layernorm"):
            for res in (False, True):
                configs.append((act, norm, res))

    grid = np.full((len(configs), len(depths)), np.nan)
    for i, (act, norm, res) in enumerate(configs):
        for j, d in enumerate(depths):
            key = (act, "he_normal", norm, res, d)
            if key in stats:
                grid[i, j] = stats[key]["accuracy"]

    fig = plt.figure(figsize=(11.6, 7.6), facecolor=CREAM)
    ax = fig.add_axes([0.235, 0.13, 0.60, 0.63])
    ax.set_facecolor(CREAM)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad(CREAM)
    im = ax.imshow(grid, cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                shade = "white" if grid[i, j] > 0.82 else INK
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        color=shade, fontsize=8.5)
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels(depths, color=MUTED, fontsize=9)
    ax.set_yticks(range(len(configs)))
    labels = [f"{a}  {'LN' if n == 'layernorm' else 'no norm'}  "
              f"{'+res' if r else ''}" for a, n, r in configs]
    ax.set_yticklabels(labels, color=INK, fontsize=8.6)
    ax.set_xlabel("depth (hidden layers)", color=MUTED, fontsize=10, labelpad=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("training accuracy", color=MUTED, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    cbar.outline.set_visible(False)

    fig.text(0.055, 0.93, "FIG. 2", color=CORAL, fontsize=10, fontweight="bold")
    fig.text(0.055, 0.885, "Where depth stops being free", color=INK,
             fontsize=19, fontweight="bold")
    fig.text(0.055, 0.845,
             f"training accuracy, He initialisation, {len(rows)} runs, 400 Adam steps each  |  "
             "pale cells are runs stuck at chance",
             color=MUTED, fontsize=8.8)
    fig.savefig(ROOT / "figures" / "depth_horizon_heatmap.png", dpi=200, facecolor=CREAM)
    print("saved figures/depth_horizon_heatmap.png")

    # ---------------------------------------------------------------- attenuation
    fig2 = plt.figure(figsize=(9.4, 6.6), facecolor=CREAM)
    ax2 = fig2.add_axes([0.115, 0.24, 0.80, 0.53])
    ax2.set_facecolor(CREAM)
    colours = {"sigmoid": CORAL, "tanh": GOLD, "relu": TEAL, "gelu": "#5B5EA6"}
    for act in ("sigmoid", "tanh", "relu", "gelu"):
        for res, style in ((False, "-"), (True, "--")):
            xs, ys = [], []
            for d in depths:
                key = (act, "he_normal", "none", res, d)
                if key in stats and np.isfinite(stats[key]["grad_ratio"]):
                    xs.append(d)
                    ys.append(stats[key]["grad_ratio"])
            ax2.plot(xs, ys, style, color=colours[act], lw=1.8,
                     marker="o", ms=4, label=f"{act}{' + residual' if res else ''}")
    ax2.set_yscale("log")
    ax2.set_xlabel("depth", color=MUTED, fontsize=10, labelpad=6)
    ax2.set_ylabel("||grad layer 1|| / ||grad layer L||", color=MUTED, fontsize=10, labelpad=6)
    ax2.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.grid(axis="y", color="#E3DCCC", lw=0.8)
    ax2.set_axisbelow(True)
    ax2.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax2.legend(frameon=False, fontsize=8, ncol=4, labelcolor=MUTED,
               loc="lower center", bbox_to_anchor=(0.45, -0.30))

    fig2.text(0.055, 0.93, "FIG. 3", color=CORAL, fontsize=10, fontweight="bold")
    fig2.text(0.055, 0.885, "What reaches the first layer", color=INK,
              fontsize=19, fontweight="bold")
    fig2.text(0.055, 0.845,
              "gradient norm at the first layer relative to the last, after the first step",
              color=MUTED, fontsize=8.8)
    fig2.savefig(ROOT / "figures" / "gradient_attenuation.png", dpi=200, facecolor=CREAM)
    print("saved figures/gradient_attenuation.png")


if __name__ == "__main__":
    main()
