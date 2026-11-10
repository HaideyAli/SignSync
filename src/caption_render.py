"""Draws the caption text onto a webcam frame.

Pillow with a real TrueType face, not cv2.putText's blocky Hershey fonts.
No panel sits behind the text — it is drawn straight onto the video, so an
outline on each glyph carries the contrast instead.

Sentence and subtitle rasterise to one cached RGBA image cropped to the
glyphs' bounding box: a caption change costs a re-render, every other frame
costs one multiply plus one add over that box.

BGR<->RGB conversion is confined to this module, so there is one place for the
channel order to be wrong rather than several.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeuib.ttf",           # semibold reads better on video
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

COL_W_FRAC    = 0.82     # text column width as a fraction of the frame
MARGIN_FRAC   = 0.045    # gap between text and the bottom edge
TEXT_COLOUR   = (245, 247, 250, 255)
SUB_COLOUR    = (150, 205, 255, 255)
# White text vanishes against a bright wall; the outline is what broadcast
# subtitles use, and it bakes into the cached raster so it costs nothing.
STROKE_COLOUR = (0, 0, 0, 205)
STROKE_FRAC   = 0.085    # outline width relative to the font size
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}
_cache: dict[tuple, tuple | None] = {}


def _font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        path = next((p for p in _FONT_CANDIDATES if Path(p).exists()), None)
        _font_cache[size] = (ImageFont.truetype(path, size) if path
                             else ImageFont.load_default())
    return _font_cache[size]


def _wrap(text: str, font, max_w: int, draw) -> list[str]:
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


def _text(draw, col_w: int, y: float, s: str, font, fill) -> None:
    """One centred, outlined line; Pillow strokes the outline itself, which
    beats drawing the string repeatedly at offsets."""
    draw.text(((col_w - draw.textlength(s, font=font)) / 2, y), s,
              font=font, fill=fill, stroke_fill=STROKE_COLOUR,
              stroke_width=max(1, round(font.size * STROKE_FRAC)))


def _fit(text: str, ladder: list[int], max_w: int, draw):
    """Largest size from the ladder that wraps to at most two lines."""
    font, lines = _font(ladder[0]), []
    for size in ladder:
        font = _font(size)
        lines = _wrap(text, font, max_w, draw)
        if len(lines) <= 2:
            break
    return font, lines[:3]


def _build(w: int, h: int, text: str, subtitle: str):
    """Rasterise the text, returning (box, premultiplied_bgr, inv_alpha)."""
    scale = h / 720.0                       # sizes tuned at 720p
    pad = int(28 * scale)
    col_w = int(w * COL_W_FRAC)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

    ladder = [max(12, int(s * scale)) for s in (44, 38, 33, 28, 24)]
    font, lines = (_fit(text, ladder, col_w - 2 * pad, probe) if text
                   else (_font(ladder[0]), []))
    sub_font = _font(max(11, int(21 * scale)))

    line_h = int(font.size * 1.28)
    sub_h = int(sub_font.size * 1.5) if subtitle else 0
    col_h = max(int(64 * scale), len(lines) * line_h + sub_h + 2 * pad)

    img = Image.new("RGBA", (col_w, col_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    y = pad
    for line in lines:
        _text(draw, col_w, y, line, font, TEXT_COLOUR)
        y += line_h
    if subtitle:
        _text(draw, col_w, y + int(4 * scale), subtitle, sub_font, SUB_COLOUR)

    # Blend only the glyphs' box: with no background most of this image is
    # transparent, and blending it would be a no-op over a large area.
    rgba = np.array(img)
    a = rgba[:, :, 3]
    rows, cols = np.flatnonzero(a.any(axis=1)), np.flatnonzero(a.any(axis=0))
    if len(rows) == 0:                           # nothing drawn
        return None
    ry0, ry1 = int(rows[0]), int(rows[-1]) + 1
    cx0, cx1 = int(cols[0]), int(cols[-1]) + 1
    sub = rgba[ry0:ry1, cx0:cx1]

    x0, y0 = (w - col_w) // 2, max(0, h - col_h - int(h * MARGIN_FRAC))
    alpha = sub[:, :, 3:4].astype(np.float32) / 255.0
    bgr = cv2.cvtColor(sub[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32)
    # Pre-multiply once so the per-frame path is a multiply plus an add. cv2
    # will not broadcast like numpy, so alpha is expanded to 3 channels here.
    inv = np.ascontiguousarray(np.repeat(1.0 - alpha, 3, axis=2))
    return ((y0 + ry0, y0 + ry1, x0 + cx0, x0 + cx1),
            np.ascontiguousarray(bgr * alpha), inv)


def _cached(w: int, h: int, text: str, subtitle: str):
    key = (w, h, text, subtitle)
    if key not in _cache:
        if len(_cache) > 24:                     # bounded: captions change constantly
            _cache.clear()
        _cache[key] = _build(w, h, text, subtitle)
    return _cache[key]


def draw_caption(frame_bgr: np.ndarray, text: str, subtitle: str = "") -> np.ndarray:
    """Composite the caption text onto the frame, in place."""
    if not text and not subtitle:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    cached = _cached(w, h, text, subtitle)
    if cached is None:                           # whitespace-only caption
        return frame_bgr
    (y0, y1, x0, x1), premult, inv_alpha = cached
    roi = frame_bgr[y0:y1, x0:x1].astype(np.float32)
    cv2.multiply(roi, inv_alpha, dst=roi)
    cv2.add(roi, premult, dst=roi)
    frame_bgr[y0:y1, x0:x1] = roi.astype(np.uint8)
    return frame_bgr
