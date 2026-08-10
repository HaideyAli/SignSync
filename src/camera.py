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

DEFAULT_EXPOSURE = -5      # 1/32s; measured 29.9 fps and twice the light of -6.
                           # -4 doubles light again but drops to 17 fps.
DEFAULT_GAIN = 220         # Setting exposure manually also disables auto-gain,
                           # leaving the sensor at minimum amplification — the
                           # cause of an almost black picture. Gain costs no
                           # frame rate at all (29.9 fps at every level), but
                           # amplifies noise with signal, so it is kept below
                           # maximum and gamma does the rest.
DEFAULT_BRIGHTNESS = 128   # This control is a black-level *offset*, so pushing
                           # it high lifts blacks and flattens the picture: at
                           # 200 the darkest 5% of pixels sat at 74/255 — grey,
                           # not black — which reads as a milky haze over the
                           # whole picture. Left at the neutral default so
                           # blacks stay black; gamma does the brightening.
DEFAULT_GAMMA = 0.80       # Brightens midtones without lifting blacks, which
                           # is what BRIGHTNESS cannot do. Measured at 1080p
                           # against b140/g200/0.55: contrast 52.6 -> 59.9 and
                           # blacks 53 -> 5, at the same grain and frame rate.
                           # Costs 2.9ms.
DSHOW_MANUAL_EXPOSURE = 0.25   # CAP_PROP_AUTO_EXPOSURE value meaning "manual"


def open_camera(index: int = 0, width: int = 0, height: int = 0,
                fps: float = 0.0, exposure: float | None = DEFAULT_EXPOSURE,
                gain: float | None = DEFAULT_GAIN,
                brightness: float | None = DEFAULT_BRIGHTNESS) -> cv2.VideoCapture:
    """Open the webcam, optionally forcing resolution/rate/exposure/gain.

    exposure=None leaves auto-exposure alone, which caps the camera at 15fps.
    Whenever exposure is set manually, gain must be set too — the driver stops
    adjusting it and the picture goes nearly black otherwise."""
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
        if gain is not None:
            cap.set(cv2.CAP_PROP_GAIN, gain)
    if brightness is not None:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
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


def gamma_lut(gamma: float | None) -> np.ndarray | None:
    """Lookup table for a gamma curve, or None for no correction.

    Applied to the *output* frame only, never to what the detector sees: this
    is a cosmetic change and the model should keep receiving the pixels its
    training data resembled.
    """
    if not gamma or gamma == 1.0:
        return None
    return np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], np.uint8)
