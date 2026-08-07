"""Capture-window state machine, extracted so the GUI worker can reuse it
without duplicating inference.py's inline loop logic (predictor.py doesn't
expose this — it only wraps the checkpoint).

Timing is not arbitrary — it mirrors how scripts/record_signs.py collected
the training data, since the model only ever saw signs in that geometry:
clips were exactly RECORD_SECS long at ~22 fps, and signing starts ~18%
into the clip and ends ~63% through. A free-running window breaks both,
which is why short signs (cool, no, yes — least active span) failed live
before this was fixed.

inference.py keeps its own inline copy of this logic for now rather than
being refactored to import this — a deliberate, separate follow-up, so
this change carries zero regression risk to that already-verified file.
"""
import time
from collections import deque

import numpy as np

from predictor import RECORD_SECS, SIGN_ONSET_FRAC, MotionTrigger

CONF_THRESHOLD = 0.80    # per CLAUDE.md
COOLDOWN_S     = 1.5
WARMUP_FRAMES  = 30      # frames used to measure real fps before sizing the window


class CaptureEngine:
    def __init__(self, seconds: float = RECORD_SECS, auto: bool = False):
        self.seconds = seconds
        self.auto = auto
        self.ring: deque = deque(maxlen=600)
        self.trigger = MotionTrigger() if auto else None
        self.prev_lm: np.ndarray | None = None
        self.window = 0
        self.need = 0
        self.frame_i = 0
        self.fps = 0.0
        self._t0 = time.time()
        self._result: list[np.ndarray] | None = None

    @property
    def ready(self) -> bool:
        """True once warmup has completed and the window is sized."""
        return self.frame_i > WARMUP_FRAMES

    @property
    def is_recording(self) -> bool:
        return self.need > 0

    @property
    def seconds_remaining(self) -> float:
        return self.need / max(self.fps, 1.0)

    def start_capture(self, pre_roll: bool = False) -> bool:
        """Begin a capture. Returns False (no-op) if already recording or
        still warming up. pre_roll=True (auto-trigger) starts the window
        before the detected motion so the sign lands ~SIGN_ONSET_FRAC into
        it; pre_roll=False (manual button) starts fresh, matching how
        record_signs.py collected the training data."""
        if self.is_recording or not self.ready:
            return False
        self.need = (max(1, self.window - int(self.window * SIGN_ONSET_FRAC))
                    if pre_roll else self.window)
        return True

    def poll_result(self) -> list[np.ndarray] | None:
        """Call once per frame after push_frame(). Returns the completed
        capture the instant it finishes, else None."""
        result, self._result = self._result, None
        return result

    def push_frame(self, lm: np.ndarray) -> None:
        """Feed one raw (258,) landmark frame. Updates fps/window sizing
        during warmup, advances the recording countdown, and auto-fires
        on sustained motion if self.auto."""
        self.ring.append(lm)
        self.frame_i += 1

        if self.frame_i <= WARMUP_FRAMES:
            self.fps = self.frame_i / max(1e-6, time.time() - self._t0)
            self.window = max(10, int(round(self.seconds * self.fps)))
        elif self.frame_i % 10 == 0:
            self.fps = 0.9 * self.fps + 0.1 * (10 / max(1e-6, time.time() - self._t0))
            self._t0 = time.time()
        if self.frame_i == WARMUP_FRAMES:
            self._t0 = time.time()

        if self.auto and self.trigger.update(self.prev_lm, lm):
            self.start_capture(pre_roll=True)
        self.prev_lm = lm

        if self.need > 0:
            self.need -= 1
            if self.need == 0 and len(self.ring) >= self.window:
                self._result = list(self.ring)[-self.window:]
