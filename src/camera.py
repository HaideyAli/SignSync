"""Webcam opening, kept separate because backend choice is a platform quirk
rather than part of the inference loop.

Measured on this machine: OpenCV's default MSMF backend takes ~19s to open the
webcam; DirectShow takes ~1.3s and yields identical (480, 640, 3) frames.
A 20-second freeze at startup reads as a crash, so prefer DirectShow on
Windows and fall back if it fails.
"""
import sys

import cv2


def open_camera(index: int = 0) -> cv2.VideoCapture:
    if sys.platform == "win32":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)
