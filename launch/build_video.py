"""Render a 30-second 1080x1080 video from the carousel pages, with intro/outro cards."""

from __future__ import annotations

import math
import os
import subprocess

from PIL import Image, ImageDraw, ImageFont

W = 1080
FPS = 30
CREAM = (247, 243, 234)
INK = (18, 22, 28)
CORAL = (217, 85, 60)
MUTED = (107, 100, 89)
TEAL = (47, 111, 115)

SERIF = "/usr/share/fonts/truetype/noto/NotoSerifDisplay-Black.ttf"
SANS = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
SANS_MED = "/usr/share/fonts/truetype/noto/NotoSans-Medium.ttf"
SANS_BOLD = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

HERE = "/home/ubuntu/video-work"
CARD = 1440


def tracked(draw, xy, text, font, fill, tracking=0.0):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + tracking
    return x


def card_intro() -> Image.Image:
    img = Image.new("RGB", (CARD, CARD), CREAM)
    d = ImageDraw.Draw(img)
    d.rectangle((110, 118, 170, 124), fill=CORAL)
    tracked(d, (190, 106), "GRADIENT LAB", ImageFont.truetype(SANS_BOLD, 30), INK, 7)

    head = ImageFont.truetype(SERIF, 148)
    d.text((104, 330), "Where depth", font=head, fill=INK)
    d.text((104, 480), "stops being", font=head, fill=INK)
    end = d.textlength("free", font=head)
    d.text((104, 630), "free", font=head, fill=INK)
    d.text((104 + end + 4, 630), ".", font=head, fill=CORAL)

    d.text((110, 876), "720 controlled runs.", font=ImageFont.truetype(SANS_MED, 46), fill=INK)
    d.text((110, 938), "One variable at a time.", font=ImageFont.truetype(SANS, 46), fill=MUTED)

    d.line((110, 1180, 1330, 1180), fill=(220, 212, 198), width=2)
    d.text((110, 1210), "github.com/imSheheryar/gradient-lab",
           font=ImageFont.truetype(MONO, 30), fill=MUTED)
    return img


def card_outro() -> Image.Image:
    img = Image.new("RGB", (CARD, CARD), CREAM)
    d = ImageDraw.Draw(img)
    d.rectangle((110, 118, 170, 124), fill=CORAL)
    tracked(d, (190, 106), "GRADIENT LAB", ImageFont.truetype(SANS_BOLD, 30), INK, 7)

    head = ImageFont.truetype(SERIF, 148)
    d.text((104, 330), "Read the claim,", font=head, fill=INK)
    d.text((104, 480), "not the setup.", font=head, fill=INK)

    d.text((110, 760), "Code, the 720-row CSV and every figure:", font=ImageFont.truetype(SANS, 42), fill=MUTED)
    d.text((110, 822), "github.com/imSheheryar/gradient-lab",
           font=ImageFont.truetype(SANS_BOLD, 42), fill=TEAL)

    d.line((110, 1180, 1330, 1180), fill=(220, 212, 198), width=2)
    d.text((110, 1210), "Sheheryar Yousaf  ·  MSc Data Science, University of Campania",
           font=ImageFont.truetype(SANS, 28), fill=MUTED)
    return img


def zoom_frame(src: Image.Image, progress: float, reverse: bool = False) -> Image.Image:
    """Center-crop with a slow zoom and return a W x W frame."""
    span = 0.07
    z = 1.0 + span * (1 - progress if reverse else progress)
    side = int(src.width / z)
    left = (src.width - side) // 2
    top = (src.height - side) // 2
    return src.crop((left, top, left + side, top + side)).resize((W, W), Image.LANCZOS)


def build_frames() -> tuple[int, list[tuple[float, int, int]]]:
    intro = card_intro()
    outro = card_outro()
    pages = [Image.open(f"{HERE}/pages/p-{i}.jpg").convert("RGB") for i in range(1, 9)]

    shots: list[tuple[Image.Image, float, bool]] = [(intro, 3.0, False)]
    for i, page in enumerate(pages):
        shots.append((page, 3.4, i % 2 == 1))
    shots.append((outro, 3.4, True))

    fade = 0.45
    # timeline: each shot starts fade-in before the previous ends
    starts = [0.0]
    for _, dur, _ in shots[:-1]:
        starts.append(starts[-1] + dur - fade)
    total = starts[-1] + shots[-1][1]

    os.makedirs(f"{HERE}/frames", exist_ok=True)
    n_frames = int(math.ceil(total * FPS))
    marks: list[tuple[float, int, int]] = []

    for f in range(n_frames):
        t = f / FPS
        # active shot = last shot whose start <= t
        idx = max(i for i, s in enumerate(starts) if s <= t + 1e-9)
        img, dur, rev = shots[idx]
        p = min(1.0, max(0.0, (t - starts[idx]) / dur))
        frame = zoom_frame(img, p, rev)

        # crossfade with the previous shot
        into = t - starts[idx]
        if idx > 0 and into < fade:
            prev_img, prev_dur, prev_rev = shots[idx - 1]
            prev_frame = zoom_frame(prev_img, 1.0, prev_rev)
            alpha = into / fade
            frame = Image.blend(prev_frame, frame, alpha)

        draw = ImageDraw.Draw(frame)
        # progress rail
        draw.rectangle((0, W - 6, W, W), fill=(232, 224, 210))
        draw.rectangle((0, W - 6, int(W * (t / total)), W), fill=CORAL)

        frame.save(f"{HERE}/frames/f{f:05d}.jpg", quality=92)
        marks.append((t, idx, f))

    return n_frames, marks


if __name__ == "__main__":
    n, _ = build_frames()
    print("frames", n)
    out = f"{HERE}/gradient_lab_walkthrough.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(FPS), "-i", f"{HERE}/frames/f%05d.jpg",
            "-c:v", "libx264", "-preset", "slow", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out,
        ],
        check=True,
    )
    print("wrote", out, os.path.getsize(out) / 1e6, "MB")
