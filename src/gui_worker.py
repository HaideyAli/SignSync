"""Camera + MediaPipe + prediction loop, running off the GUI thread.

Emits the *captioned* frame — the same pixels sent to the virtual camera — so
the preview shows what Zoom participants see. Preview emission is throttled:
a full frame per signal at camera rate outruns the UI thread and Qt's queued
connections grow unbounded. The virtual camera still gets every frame.
"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PySide6.QtCore import QThread, Signal

from camera import open_camera
from caption_render import draw_caption
from capture_engine import CaptureEngine
from extract_landmarks import HOLISTIC_MODEL_URL, ensure_model, landmarks_from_frame
from predictor import SignPredictor, hands_visible
from session import SignSession
from virtual_cam import VirtualCam

W, H = 640, 480       # must match extraction so landmark scale is consistent
PREVIEW_FPS = 15.0    # on-screen only; the virtual camera gets every frame


class InferenceWorker(QThread):
    # frames are numpy arrays — not in Qt's registered type table, so `object`
    frame_ready      = Signal(object)
    word_accepted    = Signal(str, float)
    sentence_ready   = Signal(str)
    status_changed   = Signal(str)
    vcam_status      = Signal(bool, str)   # available, device-name-or-error
    error            = Signal(str)

    def __init__(self, checkpoint_path: str, camera_index: int = 0,
                 continuous: bool = True, use_vcam: bool = True):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.camera_index = camera_index
        self.continuous = continuous
        self.use_vcam = use_vcam
        self.session = SignSession()
        self._running = False
        self._capture_requested = False
        self._clear_requested = False

    # --- called from the UI thread; single-flag writes are safe under the GIL ---
    def request_capture(self) -> None:
        self._capture_requested = True

    def request_clear(self) -> None:
        self._clear_requested = True

    def stop(self) -> None:
        self._running = False


    def run(self) -> None:
        try:
            self._run()
        except Exception as e:
            self.error.emit(str(e))

    def _run(self) -> None:
        predictor = SignPredictor(self.checkpoint_path)
        engine = CaptureEngine(auto=self.continuous)

        models_dir = Path("data/models"); models_dir.mkdir(parents=True, exist_ok=True)
        task = models_dir / "holistic_landmarker.task"
        ensure_model(HOLISTIC_MODEL_URL, task)
        opts = mp_vision.HolisticLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(task)),
            # IMAGE mode matches extraction; VIDEO/LIVE_STREAM add temporal
            # smoothing that would shift live features off the training distribution
            running_mode=mp_vision.RunningMode.IMAGE)

        self.status_changed.emit("opening camera...")
        cap = open_camera(self.camera_index)
        if not cap.isOpened():
            self.error.emit("cannot open webcam")
            return

        vcam = VirtualCam(W, H, 30.0) if self.use_vcam else None
        if vcam is not None:
            self.vcam_status.emit(vcam.available, vcam.device_name or vcam.error or "")
        last_preview, self._running = 0.0, True
        try:
            with mp_vision.HolisticLandmarker.create_from_options(opts) as detector:
                while self._running:
                    ok, frame = cap.read()
                    if not ok:
                        self.error.emit("camera read failed")
                        break

                    # Detect on the un-flipped frame — mirroring swaps hand identity
                    small = cv2.resize(frame, (W, H))
                    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    lm    = landmarks_from_frame(
                        detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)))
                    engine.push_frame(lm)

                    if self._clear_requested:
                        self._clear_requested = False
                        self.session.clear()
                    if self._capture_requested:
                        self._capture_requested = False
                        if engine.start_capture(pre_roll=False):
                            self.status_changed.emit("recording")

                    frames = engine.poll_result()
                    if frames is not None:
                        self._handle_capture(predictor, frames)

                    sentence = self.session.tick()
                    if sentence:
                        self.sentence_ready.emit(sentence)

                    out = draw_caption(cv2.flip(small, 1), self.session.caption,
                                       self._subtitle(engine))
                    if vcam is not None:
                        vcam.send(out)
                    now = time.time()
                    if now - last_preview >= 1.0 / PREVIEW_FPS:
                        last_preview = now
                        self.frame_ready.emit(out)
        finally:
            cap.release()
            if vcam is not None:
                vcam.close()

    def _subtitle(self, engine: CaptureEngine) -> str:
        if engine.is_recording:
            return f"reading sign...  {engine.seconds_remaining:.1f}s"
        if self.session.is_building:
            return self.session.subtitle
        return (self.session.rejected or
                ("watching for signs..." if self.continuous else "press Capture Sign"))

    def _handle_capture(self, predictor: SignPredictor, frames) -> None:
        if not hands_visible(frames):
            self.status_changed.emit("no hands detected")
            return
        results = predictor.predict(frames)
        accepted, message = self.session.offer(results)
        if accepted:
            self.word_accepted.emit(*results[0])
        self.status_changed.emit(message)
