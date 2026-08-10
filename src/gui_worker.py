"""Video loop, running off the GUI thread.

Emits the *captioned* frame — the same pixels sent to the virtual camera.
Detection runs on its own thread (detect_worker.py) because MediaPipe's
~42.7ms does not fit a 30fps budget; this loop only reads, tone-maps,
captions and publishes. Preview emission is throttled and downscaled, since
full frames at camera rate outrun the UI thread.
"""
import time

import cv2
from PySide6.QtCore import QThread, Signal

from camera import gamma_lut, open_camera, thumbnail
from caption_render import draw_caption
from capture_engine import CaptureEngine
from detect_worker import DetectorThread
from predictor import SignPredictor
from session import SignSession
from virtual_cam import VirtualCam

PREVIEW_FPS = 15.0    # on-screen only; the vcam gets every frame
PREVIEW_H   = 360     # preview widget is ~300px; see camera.thumbnail


class InferenceWorker(QThread):
    # frames are numpy arrays — not in Qt's registered type table, so `object`
    frame_ready      = Signal(object)
    word_accepted    = Signal(str, float)
    sentence_ready   = Signal(str)
    status_changed   = Signal(str)
    vcam_status      = Signal(bool, str)   # available, device-name-or-error
    error            = Signal(str)

    def __init__(self, checkpoint_path: str, camera_index: int = 0,
                 mode: str = "live", use_vcam: bool = True,
                 width: int = 1920, height: int = 1080, fps: float = 30.0,
                 exposure: float | None = -5.0, gain: float | None = 220.0,
                 brightness: float | None = 128.0, gamma: float | None = 0.80):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.camera_index = camera_index
        self.mode = mode                 # "live" | "cycle" | "manual"
        self.use_vcam = use_vcam
        self.width, self.height, self.fps = width, height, fps
        self.exposure, self.gain, self.bright, self.gamma = (
            exposure, gain, brightness, gamma)
        self.session = SignSession()
        self._latest = None              # newest frame, for the detector thread
        self._running = False
        self._capture_requested = False
        self._clear_requested = False
        self._cycle_request: bool | None = None

    # --- UI thread calls these; single-flag writes are safe under the GIL ---
    def request_capture(self) -> None:
        self._capture_requested = True

    def request_clear(self) -> None:
        self._clear_requested = True

    def set_running(self, on: bool) -> None:
        """Start/stop captioning — live mode streams, cycle mode prompts."""
        self._cycle_request = on

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        try:
            self._run()
        except Exception as e:
            self.error.emit(str(e))

    def _on_word(self, word: str, conf: float) -> None:
        # tick() only fires after a pause; the second emit keeps the panel
        # in step with the video caption
        self.word_accepted.emit(word, conf)
        self.sentence_ready.emit(self.session.caption)

    def _run(self) -> None:
        predictor = SignPredictor(self.checkpoint_path)
        engine = CaptureEngine()
        start, stop = ((engine.start_live, engine.stop_live) if self.mode == "live"
                       else (engine.start_cycle, engine.stop_cycle))
        self.status_changed.emit("opening camera...")
        cap = open_camera(self.camera_index, self.width, self.height,
                          self.fps, self.exposure, self.gain, self.bright)
        if not cap.isOpened():
            self.error.emit("cannot open webcam")
            return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        vcam = VirtualCam(w, h, self.fps or 30.0) if self.use_vcam else None
        if vcam is not None:
            self.vcam_status.emit(vcam.available, vcam.device_name or vcam.error or "")
        detector = DetectorThread(
            lambda: self._latest, predictor, engine, self.session,
            on_word=self._on_word, on_status=self.status_changed.emit,
            on_error=self.error.emit)
        detector.start()
        lut = gamma_lut(self.gamma)
        last_preview, self._running = 0.0, True
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    self.error.emit("camera read failed")
                    break
                self._latest = frame          # atomic rebind; detector picks it up

                if self._cycle_request is not None:
                    start() if self._cycle_request else stop()
                    self._cycle_request = None
                if self._clear_requested:
                    self._clear_requested = False
                    self.session.clear()
                if self._capture_requested:
                    self._capture_requested = False
                    engine.start_capture(pre_roll=False)

                # Display only — the detector gets the raw, un-flipped frame
                # (mirroring swaps hand identity; gamma is cosmetic)
                shown = cv2.flip(frame, 1)
                if lut is not None:
                    shown = cv2.LUT(shown, lut)
                out = draw_caption(shown, self.session.caption,
                                   self._subtitle(engine))
                if vcam is not None:
                    vcam.send(out)
                now = time.time()
                if now - last_preview >= 1.0 / PREVIEW_FPS:
                    last_preview = now
                    self.frame_ready.emit(thumbnail(out, PREVIEW_H))
        finally:
            detector.stop(); detector.join(timeout=2.0)
            cap.release()
            if vcam is not None:
                vcam.close()

    def _subtitle(self, engine: CaptureEngine) -> str:
        """Live mode shows only the signed words — the video feed is what the
        meeting sees, so no cues go on it."""
        if self.mode == "live":
            return self.session.subtitle
        if engine.is_recording:
            return f"● SIGN NOW   {engine.seconds_remaining:.1f}s"
        if engine.cycling:
            return f"get ready...  {engine.seconds_until_capture:.1f}s"
        return self.session.subtitle or self.session.rejected
