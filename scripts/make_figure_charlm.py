"""Figure 1: loss curve plus what the model writes at three points in training.

Run:  .venv/bin/python -m scripts.make_figure_charlm
"""

from __future__ import annotations

import time
import textwrap
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine import Adam, Linear, Sequential, Tanh, Tensor
from engine.ops import softmax_cross_entropy
from scripts.char_lm import build_vocab, load_corpus, make_windows, one_hot, sample

INK = "#12161C"
CREAM = "#F7F3EA"
CORAL = "#D9553C"
TEAL = "#2F6F73"
MUTED = "#6B6459"
ROOT = Path(__file__).resolve().parent.parent

STEPS = 6000
CONTEXT = 8
HIDDEN = 160
CHECKPOINTS = {0, 600, STEPS}


def trim(text: str, max_lines: int = 6, width: int = 86) -> str:
    lines: list[str] = []
    for raw in text.split("\n"):
        if not raw:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw, width) or [""])
    return "\n".join(lines[:max_lines])


def main() -> None:
    rng = np.random.default_rng(0)
    text = load_corpus()
    chars, stoi = build_vocab(text)
    vocab = len(chars)
    encoded = np.array([stoi[c] for c in text], dtype=np.int64)
    contexts, targets = make_windows(encoded, CONTEXT)

    model = Sequential(
        Linear(CONTEXT * vocab, HIDDEN, init="he_normal", rng=rng),
        Tanh(),
        Linear(HIDDEN, vocab, init="xavier_normal", rng=rng),
    )
    n_params = sum(p.size for p in model.parameters())
    opt = Adam(model.parameters(), lr=3e-3)

    history, snapshots = [], {}
    start = time.time()
    for step in range(STEPS + 1):
        if step in CHECKPOINTS:
            snapshots[step] = trim(sample(model, chars, stoi, "First Citizen:\n", 240,
                                          CONTEXT, 0.7, rng))
        idx = rng.integers(0, len(contexts), 64)
        x = Tensor(one_hot(contexts[idx], vocab))
        loss = softmax_cross_entropy(model(x), targets[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0:
            history.append((step, float(loss.data)))
    elapsed = time.time() - start

    steps = np.array([h[0] for h in history])
    losses = np.array([h[1] for h in history])
    ema = np.copy(losses)
    alpha = 0.06
    for i in range(1, len(ema)):
        ema[i] = alpha * losses[i] + (1 - alpha) * ema[i - 1]

    fig = plt.figure(figsize=(13.6, 7.4), facecolor=CREAM)

    ax = fig.add_axes([0.055, 0.13, 0.34, 0.60])
    ax.set_facecolor(CREAM)
    ax.plot(steps, losses, color=TEAL, lw=0.7, alpha=0.30)
    ax.plot(steps, ema, color=TEAL, lw=2.0)
    ax.axhline(np.log(vocab), color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(60, np.log(vocab) + 0.05, "uniform guessing (ln 65)", color=MUTED, fontsize=8)
    ax.set_xlabel("step", color=MUTED, fontsize=9, labelpad=6)
    ax.set_ylabel("cross-entropy (nats)", color=MUTED, fontsize=9, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis="y", color="#E3DCCC", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(1.2, 4.5)

    fig.text(0.055, 0.93, "FIG. 1", color=CORAL, fontsize=10, fontweight="bold")
    fig.text(0.055, 0.885, "A language model with no framework under it",
             color=INK, fontsize=18, fontweight="bold")
    fig.text(0.055, 0.848,
             f"character-level MLP  {CONTEXT * vocab} -> {HIDDEN} -> {vocab}   |   "
             f"{n_params:,} parameters   |   {STEPS} Adam steps   |   "
             f"{elapsed / STEPS * 1000:.1f} ms/step, CPU",
             color=MUTED, fontsize=8.8)

    positions = [0.775, 0.535, 0.295]
    for step, y in zip(sorted(snapshots), positions):
        fig.text(0.445, y, f"step {step}", color=CORAL, fontsize=8.5, fontweight="bold")
        fig.text(0.445, y - 0.030, snapshots[step], color=INK, fontsize=6.8,
                 family="monospace", va="top", linespacing=1.5)

    fig.savefig(ROOT / "figures" / "char_lm.png", dpi=200, facecolor=CREAM)
    print(f"saved figures/char_lm.png  final loss {losses[-1]:.3f}  "
          f"({elapsed:.1f}s, {elapsed / STEPS * 1000:.1f} ms/step)")


if __name__ == "__main__":
    main()
