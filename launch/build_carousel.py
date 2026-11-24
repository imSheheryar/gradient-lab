"""Build the LinkedIn carousel (PDF) from the six result figures.

Eight square pages: a cover, six results, and a closing summary. Cream paper,
serif headlines, one idea per page. The in-figure titles are cropped away by
launch/crop_figures.py so the slide headline is the only headline.
"""

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.image import imread
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures_cropped"
OUT = HERE / "gradient_lab_carousel.pdf"

CREAM = "#F7F3EA"
INK = "#12161C"
CORAL = "#D9553C"
TEAL = "#2F6F73"
MUTED = "#6B6459"
RULE = "#DCD3C2"

DISPLAY = "/usr/share/fonts/truetype/noto/NotoSerifDisplay-Black.ttf"
DISPLAY_BOLD = "/usr/share/fonts/truetype/noto/NotoSerifDisplay-Bold.ttf"
DISPLAY_MED = "/usr/share/fonts/truetype/noto/NotoSerifDisplay-Medium.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SANS_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

SIDE = 1080
DPI = 150
LEFT = 0.085
RIGHT = 0.915
REGION_BOTTOM = 0.135
IMG_LEFT = 0.055
IMG_RIGHT = 0.945
CAPTION_GAP = 0.034


def f(path: str, size: float) -> FontProperties:
    return FontProperties(fname=path, size=size)


def new_slide() -> plt.Figure:
    fig = plt.figure(figsize=(SIDE / DPI, SIDE / DPI), dpi=DPI)
    fig.patch.set_facecolor(CREAM)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=CREAM, edgecolor="none", zorder=-10))
    return fig


def chrome(ax, page: int, total: int) -> None:
    ax.plot([LEFT, LEFT + 0.05], [0.938, 0.938], color=CORAL, lw=1.6, solid_capstyle="butt")
    ax.text(LEFT + 0.066, 0.931, "G R A D I E N T   L A B", color=INK,
            fontproperties=f(SANS_BOLD, 6.4), va="center")
    ax.text(RIGHT, 0.931, f"{page:02d} / {total:02d}", color=MUTED,
            fontproperties=f(MONO, 6.4), va="center", ha="right")
    ax.plot([LEFT, RIGHT], [0.083, 0.083], color=RULE, lw=0.8, solid_capstyle="butt")
    ax.text(LEFT, 0.055, "github.com/imSheheryar/gradient-lab", color=MUTED,
            fontproperties=f(MONO, 6.0), va="baseline")
    ax.text(RIGHT, 0.055, "Sheheryar Yousaf", color=MUTED,
            fontproperties=f(SANS, 6.0), va="baseline", ha="right")


def place_block(fig, path: Path, region_top: float, caption: str) -> None:
    """Centre an image plus its caption inside the region above the footer."""
    img = imread(path)
    h, w = img.shape[:2]
    full_width = IMG_RIGHT - IMG_LEFT
    cap_space = CAPTION_GAP + 0.012
    avail = region_top - REGION_BOTTOM

    width = full_width
    height = width * h / w
    if height + cap_space > avail:
        height = avail - cap_space
        width = height * w / h

    top = region_top
    x0 = IMG_LEFT + (full_width - width) / 2

    ax = fig.add_axes([x0, top - height, width, height])
    ax.imshow(img)
    ax.axis("off")

    fig.axes[0].text(IMG_LEFT, top - height - CAPTION_GAP, caption, color=MUTED,
                     fontproperties=f(MONO, 6.0), va="baseline")


def content_slide(pdf, page: int, total: int, kicker: str, headline: str, body: str,
                  image: Path, caption: str) -> None:
    fig = new_slide()
    ax = fig.axes[0]
    chrome(ax, page, total)

    ax.text(LEFT, 0.862, kicker.upper(), color=CORAL, fontproperties=f(SANS_BOLD, 6.6), va="baseline")
    ax.text(LEFT - 0.002, 0.782, headline, color=INK, fontproperties=f(DISPLAY_BOLD, 21), va="baseline")

    wrapped = textwrap.fill(body, width=68)
    n_lines = wrapped.count("\n") + 1
    ax.text(LEFT, 0.716, wrapped, color=MUTED, fontproperties=f(SANS, 8.4), va="top", linespacing=1.55)

    region_top = 0.716 - 0.034 * n_lines - 0.028
    place_block(fig, image, region_top, caption)

    pdf.savefig(fig, facecolor=CREAM)
    plt.close(fig)


def cover_slide(pdf, total: int) -> None:
    fig = new_slide()
    ax = fig.axes[0]
    chrome(ax, 1, total)

    ax.text(LEFT, 0.760, "Depth doesn\u2019t", color=INK, fontproperties=f(DISPLAY, 40), va="baseline")
    ax.text(LEFT, 0.655, "kill gradients.", color=INK, fontproperties=f(DISPLAY, 40), va="baseline")
    ax.text(LEFT, 0.550, "Sigmoid does.", color=CORAL, fontproperties=f(DISPLAY, 40), va="baseline")

    ax.text(
        LEFT,
        0.440,
        "I wrote a reverse-mode autodiff engine in NumPy\nand used it as an instrument, not a demo.",
        color=MUTED,
        fontproperties=f(SANS, 9.4),
        va="top",
        linespacing=1.6,
    )

    ax.plot([LEFT, LEFT + 0.28], [0.318, 0.318], color=RULE, lw=0.9)
    for x, value, label, colour in [
        (LEFT, "720", "controlled runs", CORAL),
        (LEFT + 0.175, "41", "tests, incl. PyTorch parity", TEAL),
        (LEFT + 0.42, "0", "deep-learning frameworks", INK),
    ]:
        ax.text(x, 0.262, value, color=colour, fontproperties=f(DISPLAY_MED, 19), va="baseline")
        ax.text(x, 0.212, label, color=MUTED, fontproperties=f(SANS, 6.8), va="baseline")

    pdf.savefig(fig, facecolor=CREAM)
    plt.close(fig)


def closing_slide(pdf, total: int) -> None:
    fig = new_slide()
    ax = fig.axes[0]
    chrome(ax, total, total)

    ax.text(LEFT, 0.780, "What the numbers", color=INK, fontproperties=f(DISPLAY, 26), va="baseline")
    ax.text(LEFT, 0.708, "actually say", color=INK, fontproperties=f(DISPLAY, 26), va="baseline")

    rows = [
        ("Sigmoid dies between depth 8 and 16.", "gradient ratio 1.1e-17 at depth 32"),
        ("A residual connection buys all of it back.", "0.500 \u2192 0.994 at the same depth"),
        ("Width is not a substitute.", "128 units, still chance, depth 16+"),
        ("LayerNorm can hide the failure.", "healthy at step 0, dead by step 250"),
        ("1e-7 is a float64 claim.", "float32 tops out near 5e-5"),
    ]
    y = 0.600
    for claim, detail in rows:
        ax.plot([LEFT, LEFT + 0.022], [y + 0.012, y + 0.012], color=CORAL, lw=1.2, solid_capstyle="butt")
        ax.text(LEFT + 0.036, y + 0.012, claim, color=INK, fontproperties=f(SANS_BOLD, 8.6), va="baseline")
        ax.text(LEFT + 0.036, y - 0.022, detail, color=MUTED, fontproperties=f(MONO, 6.8), va="baseline")
        y -= 0.088

    ax.text(LEFT, 0.135, "Code, raw CSV, and the full write-up:", color=MUTED,
            fontproperties=f(SANS, 8.2), va="baseline")
    ax.text(LEFT, 0.108, "github.com/imSheheryar/gradient-lab", color=TEAL,
            fontproperties=f(MONO, 9.0), va="baseline")

    pdf.savefig(fig, facecolor=CREAM)
    plt.close(fig)


SLIDES = [
    (
        "the experiment",
        "720 runs, one variable at a time",
        "Four activations, three initialisations, normalisation and residuals on or off, five depths, three seeds. Nothing varies except the thing being tested.",
        "depth_horizon_heatmap.png",
        "fig. 2 \u2014 training accuracy; pale cells never leave chance",
    ),
    (
        "measurement",
        "What reaches the first layer",
        "The gradient at layer one divided by the gradient at the last. A straight line on a log axis is exponential decay per layer.",
        "gradient_attenuation.png",
        "fig. 3 \u2014 gradient ratio after the first backward pass",
    ),
    (
        "the control",
        "Width is not depth",
        "A sigmoid net at depth 8 needs 16 units. At depth 16 and 32 it never leaves chance at any width up to 128. ReLU needs 8 at every depth.",
        "width_vs_depth.png",
        "fig. 4 \u2014 minimum width that reaches 0.90 accuracy",
    ),
    (
        "the counterexample",
        "A failure that looks healthy at step zero",
        "ReLU + LayerNorm at depth 32 starts at a healthy 0.16 and is at 1.5e-6 by step 250, accuracy stuck at 0.500.",
        "probe_trajectories.png",
        "fig. 6 \u2014 gradient ratio per training step, four runs",
    ),
    (
        "the caveat",
        "1e-7 is a float64 claim",
        "The same check at float32 tops out at 5.2e-5, and the analytical gradient already differs from the float64 truth by 8.8e-8.",
        "precision.png",
        "fig. 5 \u2014 finite-difference error against step size",
    ),
    (
        "it also learns",
        "A character-level language model",
        "Eight characters of context, one hidden layer, 93,825 parameters, 6,000 Adam steps at 3.1 ms per step. Loss 4.24 \u2192 1.93 nats.",
        "char_lm.png",
        "fig. 1 \u2014 loss and samples from the same engine",
    ),
]


def build() -> Path:
    total = len(SLIDES) + 2
    with PdfPages(OUT) as pdf:
        cover_slide(pdf, total)
        for i, (kicker, headline, body, image, caption) in enumerate(SLIDES, start=2):
            content_slide(pdf, i, total, kicker, headline, body, FIG / image, caption)
        closing_slide(pdf, total)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
