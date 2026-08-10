"""Webcam opening and framing, kept separate because backend and exposure
quirks are platform details rather than part of the inference loop.

Two measured facts drive this file:

* MSMF (OpenCV's Windows default) takes ~19s to open the camera; DirectShow
  takes ~1.3s for identical frames. A 20-second freeze reads as a crash.
* Auto-exposure, not bandwidth, caps the frame rate. Every resolution
  measured exactly 66ms/frame — the constant across 480p/720p/1080p gave it
  away as shutter time. Exposure -4 is 1/16s = 62.5ms. Forcing manual
  exposure yields 30fps, including at 1920x1080 (29.6 measured).
"""
import sys

import cv2
import numpy as np

DEFAULT_EXPOSURE = -6      # 1/64s; measured 30.1 fps. -4 (auto) gives 15 fps.
DSHOW_MANUAL_EXPOSURE = 0.25   # CAP_PROP_AUTO_EXPOSURE value meaning "manual"


def open_camera(index: int = 0, width: int = 0, height: int = 0,
                fps: float = 0.0, exposure: float | None = DEFAULT_EXPOSURE
                ) -> cv2.VideoCapture:
    """Open the webcam, optionally forcing resolution/rate/exposure.

    exposure=None leaves auto-exposure alone, which is correct in dim rooms
    but caps the camera at 15fps."""
    cap = None
    if sys.platform == "win32":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = None
    if cap is None:
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return cap

    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    if exposure is not None:
        # Order matters: auto must be disabled before a manual value sticks
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, DSHOW_MANUAL_EXPOSURE)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    return cap


def crop_4_3(frame: np.ndarray) -> np.ndarray:
    """Centre-crop to 4:3 for the detector.

    Training clips are 4:3 — the webcam is 640x480 natively and
    record_signs.py resized to 640x480, so the recordings carry no aspect
    distortion. Squashing a 16:9 frame straight to 640x480 would stretch
    landmarks horizontally by 1.33x, and normalise_landmarks divides by
    shoulder width (a scalar), so it cannot undo an anisotropic stretch.
    Cropping first keeps live geometry identical to training.
    """
    h, w = frame.shape[:2]
    target_w = int(round(h * 4 / 3))
    if target_w >= w:
        return frame                       # already 4:3 or taller
    x0 = (w - target_w) // 2
    return frame[:, x0:x0 + target_w]


def thumbnail(frame: np.ndarray, height: int) -> np.ndarray:
    """Downscale for the on-screen preview. Shipping full 1080p frames to the
    UI queues ~6MB each and makes the UI thread rescale them."""
    scale = height / frame.shape[0]
    if scale >= 1:
        return frame
    return cv2.resize(frame, (int(frame.shape[1] * scale), height))
