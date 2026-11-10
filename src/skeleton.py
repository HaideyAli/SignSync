"""Keeps the drawn skeleton in step with the hand.

detect_worker runs MediaPipe on its own thread at roughly half the video rate,
so a landmark frame is 40-90ms old by the time the renderer sees it. Drawn
as-is it visibly trails the hand — the "slow skeleton" — and the faster the
sign, the further behind it sits.

Velocity between the last two detections extrapolates across that gap, which
is what AR overlays do rather than accepting the sensor's latency. The lead is
capped so a stalled or dropped detection parks the skeleton instead of flinging
it off screen.

Cosmetic only, and deliberately kept out of the detector's path: the
classifier is fed the raw measurements from detect_worker and never these
predicted ones, so nothing here can move the model off its training
distribution (the same reasoning holistic.py gives for RunningMode.IMAGE).
"""
import time

import numpy as np

from overlay import draw_landmarks

MAX_LEAD_S = 0.10     # never predict further ahead than this
_HANDS = ((0, 63), (63, 126))
_POSE = 126
_CLAMP = (-0.5, 1.5)  # normalised coords; real ones sit in 0..1


class SkeletonTracker:
    """Wraps draw_landmarks with velocity prediction across detector lag."""

    def __init__(self, max_lead: float = MAX_LEAD_S):
        self.max_lead = max_lead
        self._prev: tuple[np.ndarray, float] | None = None
        self._last: tuple[np.ndarray, float] | None = None
        self._seen: np.ndarray | None = None

    def draw(self, frame, lm: np.ndarray | None, mirrored: bool = True):
        """Draw the predicted skeleton. `lm` is detect_worker.latest; the same
        array arriving twice means no new detection, so it is not re-timed."""
        now = time.perf_counter()
        if lm is not None and lm is not self._seen:
            self._seen = lm
            self._prev, self._last = self._last, (lm, now)
        if self._last is None:
            return frame
        return draw_landmarks(frame, self._predict(now), mirrored)

    def _predict(self, now: float) -> np.ndarray:
        lm, t = self._last
        if self._prev is None:
            return lm
        prev, t_prev = self._prev
        dt = t - t_prev
        lead = min(now - t, self.max_lead)
        if dt <= 1e-4 or lead <= 0.0:
            return lm

        out = lm.copy()
        step = (lm - prev) * (lead / dt)
        # Extrapolate a block only when it was present in BOTH frames. A hand
        # that just appeared has `prev` all zeros, so the velocity would be the
        # full landmark vector and would hurl the skeleton across the frame.
        for a, b in _HANDS:
            if prev[a:b].any() and lm[a:b].any():
                out[a:b] += step[a:b]
        if prev[_POSE:].any() and lm[_POSE:].any():
            pose_step = step[_POSE:].copy()
            pose_step[3::4] = 0.0        # visibility is a probability, not a position
            out[_POSE:] += pose_step
        return np.clip(out, *_CLAMP)
