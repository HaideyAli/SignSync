"""Camera + MediaPipe + prediction loop, running off the GUI thread.
No sentence assembly or Zoom logic here — see sentence_engine.py /
zoom_bridge.py, both called from gui_app.py on the UI thread.
"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PySide6.QtCore import QThread, Signal

from capture_engine import CaptureEngine, CONF_THRESHOLD, COOLDOWN_S
from extract_landmarks import HOLISTIC_MODEL_URL, ensure_model, landmarks_from_frame
from predictor import SignPredictor, hands_visible

W, H = 640, 480   # must match extraction so landmark scale is consistent


class InferenceWorker(QThread):
    # frame is a numpy array — not in Qt's registered type table, so `object`
    frame_ready      = Signal(object)
    prediction_ready = Signal(str, float, list)   # word, confidence, [(alt_word, alt_conf), ...]
    status_changed   = Signal(str)
    error            = Signal(str)

    def __init__(self, checkpoint_path: str, camera_index: int = 0, auto: bool = False):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.camera_index = camera_index
        self.auto = auto
        self._running = False
        self._capture_requested = False

    def request_capture(self) -> None:
        """UI thread sets this bool; the worker reads it once per loop
        iteration. A single-bool write/read is safe under the GIL without
        extra locking — documented rather than silently relied on."""
        self._capture_requested = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        try:
            self._run()
        except Exception as e:
            self.error.emit(str(e))

    def _run(self) -> None:
        predictor = SignPredictor(self.checkpoint_path)
        engine = CaptureEngine(auto=self.auto)

        models_dir = Path("data/models"); models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "holistic_landmarker.task"
        ensure_model(HOLISTIC_MODEL_URL, model_path)

        opts = mp_vision.HolisticLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            # IMAGE mode matches extraction; VIDEO/LIVE_STREAM add temporal
            # smoothing that would shift live features off the training distribution
            running_mode=mp_vision.RunningMode.IMAGE)

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error.emit("cannot open webcam")
            return

        last_emit = 0.0
        self._running = True
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

                    if self._capture_requested:
                        self._capture_requested = False
                        if engine.start_capture(pre_roll=False):
                            self.status_changed.emit("recording")

                    frames = engine.poll_result()
                    if frames is not None:
                        last_emit = self._handle_capture(predictor, frames, last_emit)
                    elif engine.is_recording:
                        self.status_changed.emit(f"recording  {engine.seconds_remaining:.1f}s")

                    # .copy() — cv2.flip's output backs a buffer the next
                    # iteration will overwrite before Qt has rendered it
                    self.frame_ready.emit(cv2.flip(small, 1).copy())
        finally:
            cap.release()

    def _handle_capture(self, predictor: SignPredictor, frames, last_emit: float) -> float:
        if not hands_visible(frames):
            self.status_changed.emit("no hands detected")
            return last_emit

        results = predictor.predict(frames)
        word, conf = results[0]
        now = time.time()
        if conf >= CONF_THRESHOLD and now - last_emit >= COOLDOWN_S:
            self.prediction_ready.emit(word, conf, results[1:3])
            self.status_changed.emit("idle")
            return now

        self.status_changed.emit(f"not confident: {word} {conf*100:.0f}%")
        return last_emit
