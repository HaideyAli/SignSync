"""Publishes captioned frames as a webcam device Zoom can select.

pyvirtualcam needs a backend driver; on Windows that means OBS Studio
installed once (its installer registers the virtual-camera device — OBS
itself does not need to be running). With no backend it raises RuntimeError
naming the problem, which this wrapper captures into `.error` so the UI can
show it instead of the app dying.
"""
import cv2
import numpy as np

try:
    import pyvirtualcam
except ImportError:                                  # dependency not installed at all
    pyvirtualcam = None

INSTALL_HINT = "No virtual camera found — install OBS Studio (free), then restart."


class VirtualCam:
    """Wraps pyvirtualcam so callers never have to care whether it exists."""

    def __init__(self, width: int, height: int, fps: float = 30.0):
        self._cam = None
        self.error: str | None = None

        if pyvirtualcam is None:
            self.error = "pyvirtualcam not installed — pip install pyvirtualcam"
            return
        try:
            self._cam = pyvirtualcam.Camera(width=width, height=height, fps=int(fps))
        except RuntimeError:
            # Backend missing is the overwhelmingly common case; the library's
            # own message is long and mentions both backends, so use our hint
            self.error = INSTALL_HINT
        except Exception as e:
            self.error = f"virtual camera failed: {e}"

    @property
    def available(self) -> bool:
        return self._cam is not None

    @property
    def device_name(self) -> str | None:
        return self._cam.device if self._cam else None

    def send(self, frame_bgr: np.ndarray) -> None:
        """No-op when unavailable, so the app runs window-only without branching."""
        if self._cam is None:
            return
        self._cam.send(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    def close(self) -> None:
        if self._cam is not None:
            self._cam.close()
            self._cam = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
