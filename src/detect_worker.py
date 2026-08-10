"""Landmark detection loop, run on its own thread.

MediaPipe costs ~42.7ms per frame, which cannot fit inside the 33.3ms budget
of a 30fps video loop. Running it here lets the video path hit 30fps for the
virtual camera while detection proceeds at whatever rate it manages (~23fps),
dropping frames when it falls behind.

Dropping frames is correct rather than lossy: CaptureEngine sizes its window
from its own measured rate, and the segmenter's thresholds are durations, so
a detection rate that differs from the video rate stays right in seconds.
"""
import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision

from camera import crop_4_3
from extract_landmarks import landmarks_from_frame
from holistic import holistic_options
from predictor import hands_visible

W, H = 640, 480      # detector input; must match extraction


class DetectorThread(threading.Thread):
    """Consumes the newest frame from `get_frame` and feeds the engine."""

    def __init__(self, get_frame, predictor, engine, session,
                 on_word=None, on_status=None, on_error=None):
        super().__init__(daemon=True)
        self.get_frame = get_frame
        self.predictor = predictor
        self.engine = engine
        self.session = session
        self.on_word = on_word or (lambda *_: None)
        self.on_status = on_status or (lambda *_: None)
        self.on_error = on_error or (lambda *_: None)
        self.running = False
        self.fps = 0.0

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        try:
            self._loop()
        except Exception as e:                      # never kill the video path
            self.on_error(f"detector: {e}")

    def _loop(self) -> None:
        self.running = True
        last_obj, n, t0 = None, 0, time.time()
        with mp_vision.HolisticLandmarker.create_from_options(holistic_options()) as det:
            while self.running:
                frame = self.get_frame()
                if frame is None or frame is last_obj:
                    time.sleep(0.004)               # nothing new yet
                    continue
                last_obj = frame

                # Crop to 4:3 before downscaling — a straight 16:9 squash
                # stretches landmarks 25% horizontally (see camera.crop_4_3)
                small = cv2.resize(crop_4_3(frame), (W, H))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                lm = landmarks_from_frame(
                    det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)))
                self.engine.push_frame(lm)

                window = self.engine.poll_result()
                if window is not None:
                    self._classify(window)
                self.session.tick()

                n += 1
                if n % 10 == 0:
                    now = time.time()
                    self.fps = 10 / max(1e-6, now - t0)
                    t0 = now

    def _classify(self, frames) -> None:
        if not hands_visible(frames):
            self.on_status("no hands detected")
            return
        results = self.predictor.predict(frames)
        accepted, message = self.session.offer(results)
        if accepted:
            self.on_word(*results[0])
        self.on_status(message)
