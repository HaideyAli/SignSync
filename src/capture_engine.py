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
REFRACTORY_S   = 1.2     # ignore triggers just after a capture — the hand dropping
                         # back to rest is itself motion and would re-fire instantly
READY_GAP_S    = 2.0     # "get ready" pause between captures in cycle mode


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
        self._refractory_until = 0.0
        self.cycling = False
        self._next_capture_at = 0.0

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

    # --- cycle mode -------------------------------------------------------
    # Motion-onset detection proved unusable here: the running baseline spans
    # ~6.6s at the real loop rate, so it climbs while you sign and the trigger
    # fires as the sign *ends*. Signs with little motion (yes ~1.6s, no ~1.5s,
    # cool ~0.9s) were missed entirely. A fixed rhythm removes the guesswork
    # and reproduces the recording geometry: capture starts on the cue, and
    # natural reaction time puts the sign ~18% in, exactly as in training.

    def start_cycle(self) -> None:
        self.cycling = True
        self._next_capture_at = time.time() + READY_GAP_S

    def stop_cycle(self) -> None:
        self.cycling = False

    @property
    def seconds_until_capture(self) -> float:
        if not self.cycling or self.is_recording:
            return 0.0
        return max(0.0, self._next_capture_at - time.time())

    def start_capture(self, pre_roll: bool = False) -> bool:
        """Begin a capture. Returns False (no-op) if already recording or
        still warming up. pre_roll=True (auto-trigger) starts the window
        before the detected motion so the sign lands ~SIGN_ONSET_FRAC into
        it; pre_roll=False (manual button) starts fresh, matching how
        record_signs.py collected the training data."""
        if self.is_recording or not self.ready or time.time() < self._refractory_until:
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

        # Time from the first *frame*, not from construction: the camera takes
        # ~1.3s to open and MediaPipe's first inferences are slow, so timing
        # from __init__ underestimates fps badly and yields a window far
        # shorter than RECORD_SECS — which silently misaligns every capture.
        if self.frame_i == 1:
            self._t0 = time.time()
            return
        if self.frame_i <= WARMUP_FRAMES:
            self.fps = (self.frame_i - 1) / max(1e-6, time.time() - self._t0)
        elif self.frame_i % 10 == 0:
            self.fps = 0.9 * self.fps + 0.1 * (10 / max(1e-6, time.time() - self._t0))
            self._t0 = time.time()
        if self.frame_i == WARMUP_FRAMES:
            self._t0 = time.time()

        # Re-size between captures so the window always spans `seconds` of real
        # time; MediaPipe keeps speeding up for hundreds of frames after start.
        if not self.is_recording:
            self.window = max(10, int(round(self.seconds * self.fps)))

        if self.auto and self.trigger.update(self.prev_lm, lm):
            self.start_capture(pre_roll=True)
        self.prev_lm = lm

        if (self.cycling and not self.is_recording and self.ready
                and time.time() >= self._next_capture_at):
            self.start_capture(pre_roll=False)

        if self.need > 0:
            self.need -= 1
            if self.need == 0 and len(self.ring) >= self.window:
                self._result = list(self.ring)[-self.window:]
                now = time.time()
                self._refractory_until = now + REFRACTORY_S
                self._next_capture_at = now + READY_GAP_S
