"""Render the 1200x627 cover image for the LinkedIn post.

Cream paper, ink type, one coral word doing the work.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

CREAM = "#F7F3EA"
INK = "#12161C"
CORAL = "#D9553C"
TEAL = "#2F6F73"
MUTED = "#6B6459"
RULE = "#DCD3C2"

DISPLAY = "/usr/share/fonts/truetype/noto/NotoSerifDisplay-Black.ttf"
DISPLAY_MED = "/usr/share/fonts/truetype/noto/NotoSerifDisplay-Medium.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SANS_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

W, H = 1200, 627
OUT = Path(__file__).resolve().parent / "cover_1200x627.png"


def font(path: str, size: float) -> FontProperties:
    return FontProperties(fname=path, size=size)


def spaced(text: str) -> str:
    return "\u2009".join(text)


def build() -> Path:
    fig = plt.figure(figsize=(W / 200, H / 200), dpi=200)
    fig.patch.set_facecolor(CREAM)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=CREAM, edgecolor="none", zorder=-10))

    left = 0.072

    # top rule and kicker
    ax.plot([left, left + 0.055], [0.905, 0.905], color=CORAL, lw=1.6, solid_capstyle="butt")
    ax.text(
        left + 0.075,
        0.898,
        spaced("GRADIENT LAB"),
        color=INK,
        fontproperties=font(SANS_BOLD, 7.2),
        va="center",
    )
    ax.text(
        1 - left,
        0.898,
        spaced("720 RUNS  \u00b7  NUMPY ONLY"),
        color=MUTED,
        fontproperties=font(MONO, 6.6),
        va="center",
        ha="right",
    )

    # headline
    ax.text(left - 0.004, 0.742, "Depth doesn\u2019t", color=INK, fontproperties=font(DISPLAY, 34), va="baseline")
    ax.text(left - 0.004, 0.596, "kill gradients.", color=INK, fontproperties=font(DISPLAY, 34), va="baseline")
    ax.text(left - 0.004, 0.450, "Sigmoid does.", color=CORAL, fontproperties=font(DISPLAY, 34), va="baseline")

    # supporting line
    ax.text(
        left,
        0.338,
        "A sigmoid network at depth 32 passes 1.1e-17 of the gradient to its first layer.",
        color=MUTED,
        fontproperties=font(SANS, 9.6),
        va="baseline",
    )
    ax.text(
        left,
        0.288,
        "One skip connection restores it. Sixteen times the width does not.",
        color=MUTED,
        fontproperties=font(SANS, 9.6),
        va="baseline",
    )

    # rule
    ax.plot([left, 1 - left], [0.228, 0.228], color=RULE, lw=0.9, solid_capstyle="butt")

    # stat strip
    stats = [
        ("1.1e-17", "gradient at layer 1", CORAL),
        ("0.994", "accuracy with residual", TEAL),
        ("16\u00d7", "width buys nothing", INK),
        ("41", "tests, no frameworks", INK),
    ]
    xs = [left, left + 0.232, left + 0.462, left + 0.690]
    for x, (value, label, colour) in zip(xs, stats):
        ax.text(x, 0.156, value, color=colour, fontproperties=font(DISPLAY_MED, 17), va="baseline")
        ax.text(x, 0.108, label, color=MUTED, fontproperties=font(SANS, 6.9), va="baseline")

    # footer
    ax.text(
        left,
        0.045,
        "github.com/imSheheryar/gradient-lab",
        color=INK,
        fontproperties=font(MONO, 7.4),
        va="baseline",
    )
    ax.text(
        1 - left,
        0.045,
        "Sheheryar Yousaf",
        color=MUTED,
        fontproperties=font(SANS, 7.4),
        va="baseline",
        ha="right",
    )

    fig.savefig(OUT, dpi=200, facecolor=CREAM)
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
