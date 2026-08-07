"""Draws the caption bar onto a webcam frame.

Uses Pillow with a real TrueType face rather than cv2.putText's Hershey
fonts, which look blocky on a video call.

Text is rasterised to a transparent overlay and cached, because a full
numpy->PIL->numpy round-trip per frame at 30fps saturates the capture
thread. Only a caption *change* costs a re-render; steady-state frames pay
one darken plus one alpha composite over the bar region.

All BGR<->RGB conversion is confined to this module so there is one place
for the channel order to be wrong, not several.
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
_overlay_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


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


def _build_overlay(w: int, bar_h: int, text: str, subtitle: str):
    """Rasterise the caption text once into (bgr, alpha) strips of the bar."""
    img = Image.new("RGBA", (w, bar_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin, max_w = int(w * 0.05), w - 2 * int(w * 0.05)

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
    sub_h  = (sub_font.size + 6) if subtitle else 0
    y = max(6, (bar_h - (len(lines) * line_h + sub_h)) // 2)

    for line in lines:
        draw.text(((w - draw.textlength(line, font=font)) / 2, y), line,
                  font=font, fill=(255, 255, 255, 255))
        y += line_h
    if subtitle:
        draw.text(((w - draw.textlength(subtitle, font=sub_font)) / 2, y), subtitle,
                  font=sub_font, fill=(150, 158, 170, 255))

    rgba = np.array(img)
    bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    alpha = (rgba[:, :, 3:4].astype(np.float32) / 255.0)
    return bgr, alpha


def _overlay(w: int, bar_h: int, text: str, subtitle: str):
    key = (w, bar_h, text, subtitle)
    if key not in _overlay_cache:
        if len(_overlay_cache) > 32:          # bounded: captions change constantly
            _overlay_cache.clear()
        _overlay_cache[key] = _build_overlay(w, bar_h, text, subtitle)
    return _overlay_cache[key]


def draw_caption(frame_bgr: np.ndarray, text: str, subtitle: str = "") -> np.ndarray:
    """Return a copy of the frame with a translucent caption bar at the bottom."""
    if not text and not subtitle:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    bar_h = int(h * BAR_HEIGHT_FRAC)
    out = frame_bgr.copy()
    roi = out[h - bar_h:h]

    # Darken the bar region (cheap, in-place)
    dark = np.full_like(roi, (12, 14, 18))
    cv2.addWeighted(dark, BAR_ALPHA, roi, 1 - BAR_ALPHA, 0, dst=roi)

    # Composite the cached text raster over it
    text_bgr, alpha = _overlay(w, bar_h, text, subtitle)
    roi[:] = (roi * (1 - alpha) + text_bgr * alpha).astype(np.uint8)
    return out
