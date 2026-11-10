"""Publishes captioned frames as a webcam device Zoom can select.

pyvirtualcam needs a backend driver; on Windows that means OBS Studio
installed once (its installer registers the virtual-camera device — OBS
itself does not need to be running). With no backend it raises RuntimeError
naming the problem, which this wrapper captures into `.error` so the UI can
show it instead of the app dying.

Publishing at 1080p was measured as the single most expensive stage in the
whole video loop — 12.1ms per frame, against ~6ms for everything else
combined. It dropped the loop to 21.8fps and, worse, starved the detector
thread from 17.2fps to 10.4fps, because both compete for the same core.
Two measured changes remove ~9ms of that:

* Publish at 720p (12.1ms -> 3.9ms). Zoom re-encodes to its own bitrate
  anyway, so the 1080p frames were being thrown away downstream.
* Ask pyvirtualcam for BGR (3.9ms -> 2.9ms) and hand it OpenCV's native
  byte order, deleting a full-frame cvtColor per frame.
"""
import cv2
import numpy as np

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
except ImportError:                                  # dependency not installed at all
    pyvirtualcam = None
    PixelFormat = None

INSTALL_HINT = "No virtual camera found — install OBS Studio (free), then restart."
MAX_OUTPUT_H = 720      # cap on the published height; capture stays full-res


class VirtualCam:
    """Wraps pyvirtualcam so callers never have to care whether it exists."""

    def __init__(self, width: int, height: int, fps: float = 30.0):
        self._cam = None
        self._bgr = False
        self.error: str | None = None

        if pyvirtualcam is None:
            self.error = "pyvirtualcam not installed — pip install pyvirtualcam"
            return

        self.width, self.height = self._output_size(width, height)
        try:
            self._open(int(fps))
        except RuntimeError:
            # Backend missing is the overwhelmingly common case; the library's
            # own message is long and mentions both backends, so use our hint
            self.error = INSTALL_HINT
        except Exception as e:
            self.error = f"virtual camera failed: {e}"

    @staticmethod
    def _output_size(width: int, height: int) -> tuple[int, int]:
        """Scale down to MAX_OUTPUT_H, keeping aspect and even dimensions
        (odd sizes break the YUV conversion in some backends)."""
        if height <= MAX_OUTPUT_H:
            return width, height
        scale = MAX_OUTPUT_H / height
        return (int(round(width * scale)) & ~1, MAX_OUTPUT_H & ~1)

    def _open(self, fps: int) -> None:
        """Prefer BGR so send() can pass OpenCV frames straight through;
        fall back to RGB on builds of pyvirtualcam that lack the format."""
        try:
            self._cam = pyvirtualcam.Camera(width=self.width, height=self.height,
                                            fps=fps, fmt=PixelFormat.BGR)
            self._bgr = True
        except (AttributeError, TypeError, ValueError):
            self._cam = pyvirtualcam.Camera(width=self.width, height=self.height, fps=fps)
            self._bgr = False

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
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height),
                                   interpolation=cv2.INTER_AREA)
        self._cam.send(frame_bgr if self._bgr
                       else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    def close(self) -> None:
        if self._cam is not None:
            self._cam.close()
            self._cam = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
