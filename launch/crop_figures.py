"""Strip the in-figure titles and dead margin from the result figures.

The carousel supplies its own headline, so the title baked into each PNG is
redundant; cropping it also makes the plot itself larger on the slide.
"""

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "figures"
OUT = Path(__file__).resolve().parent / "figures_cropped"

CREAM = np.array([247, 243, 234])


MANUAL_TOP = {"char_lm.png": 0.150}


def crop(path: Path, long_run: float = 0.45) -> Image.Image:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(int)
    mask = (np.abs(arr - CREAM).max(axis=2) > 12)

    rows = mask.sum(axis=1)
    threshold = long_run * arr.shape[1]
    candidates = np.where(rows > threshold)[0]
    top = int(candidates[0]) if len(candidates) else 0

    cols = mask.sum(axis=0)
    col_hits = np.where(cols > 0)[0]
    row_hits = np.where(mask.any(axis=1))[0]

    pad = 10
    left = max(0, int(col_hits[0]) - pad)
    right = min(arr.shape[1], int(col_hits[-1]) + pad + 1)
    bottom = min(arr.shape[0], int(row_hits[-1]) + pad + 1)
    if path.name in MANUAL_TOP:
        top = int(MANUAL_TOP[path.name] * arr.shape[0])
    top = max(0, top - 24)

    return img.crop((left, top, right, bottom))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for path in sorted(SRC.glob("*.png")):
        im = crop(path)
        target = OUT / path.name
        im.save(target)
        print(f"{path.name:32s} {Image.open(path).size} -> {im.size}")


if __name__ == "__main__":
    main()
