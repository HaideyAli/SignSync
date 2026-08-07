"""Draws the caption bar onto a webcam frame.

Uses Pillow with a real TrueType face rather than cv2.putText's Hershey
fonts, which look blocky on a video call. All BGR<->RGB conversion is
confined to this module so there is one place for the channel order to be
wrong, not several.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

BAR_HEIGHT_FRAC = 0.26     # bottom slice of the frame the caption bar occupies
BAR_ALPHA       = 0.62
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        for path in _FONT_CANDIDATES:
            if Path(path).exists():
                _font_cache[size] = ImageFont.truetype(path, size)
                break
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: int, draw: ImageDraw.ImageDraw) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_caption(frame_bgr: np.ndarray, text: str, subtitle: str = "") -> np.ndarray:
    """Return a copy of the frame with a translucent caption bar at the bottom.

    Shrinks the font and wraps until the text fits the bar, so a long
    sentence degrades gracefully instead of overflowing off-frame."""
    if not text and not subtitle:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    bar_h = int(h * BAR_HEIGHT_FRAC)
    out = frame_bgr.copy()

    # Translucent bar via real alpha blending over the existing pixels
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (12, 14, 18), thickness=-1)
    cv2.addWeighted(overlay, BAR_ALPHA, out, 1 - BAR_ALPHA, 0, dst=out)

    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    margin = int(w * 0.05)
    max_w = w - 2 * margin

    lines, font = [], _font(30)
    if text:
        for size in (30, 26, 22, 19, 16):
            font = _font(size)
            lines = _wrap(text, font, max_w, draw)
            if len(lines) <= 2:
                break
        lines = lines[:3]

    sub_font = _font(15)
    line_h = (font.size + 8) if lines else 0
    sub_h = (sub_font.size + 6) if subtitle else 0
    block_h = len(lines) * line_h + sub_h
    y = h - bar_h + max(6, (bar_h - block_h) // 2)

    for line in lines:
        x = (w - draw.textlength(line, font=font)) / 2
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        y += line_h

    if subtitle:
        x = (w - draw.textlength(subtitle, font=sub_font)) / 2
        draw.text((x, y), subtitle, font=sub_font, fill=(150, 158, 170))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
