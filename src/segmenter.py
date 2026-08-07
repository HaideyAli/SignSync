"""Finds where one sign starts and ends in a continuous stream.

Why segment at all: the model has 50 classes and no "not a sign" class, so it
labels whatever it is shown. Scoring a free-running window every few frames
therefore produces confident nonsense whenever the window straddles two signs
— measured up to 0.965 confidence with 0.954 margin, *higher* than the
weakest genuine word (0.830/0.700). No confidence or consensus threshold can
separate those, so the fix is to evaluate once per sign instead of
continuously.

The earlier MotionTrigger failed because its baseline was a rolling median
over all recent frames, so it climbed while you signed and fired as the sign
ended. Here the rest level only updates *while at rest*, and entry/exit use
different thresholds (hysteresis) so a mid-sign dip does not split a sign.

Tuned against real recordings streamed back-to-back: 7/7 sequences segmented
and classified exactly, one prediction per sign.
"""
import json
from pathlib import Path

import numpy as np

CALIB_PATH = Path("data/motion_calib.json")

ENTER_MULT   = 2.0     # motion above rest*this starts a sign
EXIT_MULT    = 1.5     # ...and must fall below rest*this to end it
QUIET_FRAMES = 25      # sustained quiet needed to close a sign (~1.7s at 15fps).
                       # Shorter values split a sign at mid-sign pauses.
MIN_LEN      = 12      # shorter bursts are twitches, not signs
MAX_LEN      = 90      # force-close after ~6s so a sign always ends
FLOOR        = 0.15    # absolute floor so a perfectly still baseline can't
                       # make the entry threshold vanish


def load_calibration() -> dict:
    """Thresholds measured on the actual camera by scripts/calibrate_motion.py.

    The defaults here were tuned against stored clips, where rest is a frozen
    frame with exactly zero motion — nothing like a live camera, where resting
    hands jitter. Run the calibration script if signs are missed or invented."""
    try:
        return json.loads(CALIB_PATH.read_text())
    except (FileNotFoundError, ValueError):
        return {}
SIGN_ONSET_FRAC  = 0.18   # measured: signing starts 18% into a training clip
SIGN_ACTIVE_FRAC = 0.45   # ...and occupies ~45% of it


def hand_motion(prev: np.ndarray, cur: np.ndarray) -> float:
    """Total hand movement between frames; 0 when either frame has no hands."""
    if prev is None or np.abs(cur[:126]).sum() < 1e-6 or np.abs(prev[:126]).sum() < 1e-6:
        return 0.0
    return float(np.abs(cur[:126] - prev[:126]).sum())


class SignSegmenter:
    """Feed frames; get back (start, end) indices the moment a sign completes."""

    def __init__(self, enter: float | None = None, exit_: float = EXIT_MULT,
                 quiet_frames: int = QUIET_FRAMES, min_len: int = MIN_LEN,
                 max_len: int = MAX_LEN, floor: float | None = None):
        calib = load_calibration()
        enter = enter if enter is not None else calib.get("enter_mult", ENTER_MULT)
        floor = floor if floor is not None else calib.get("floor", FLOOR)
        self.enter, self.exit_ = enter, exit_
        self.quiet_frames, self.min_len, self.max_len = quiet_frames, min_len, max_len
        self.floor = floor
        self.rest: float | None = None
        self.in_sign = False
        self.quiet = 0
        self.peak = 0.0
        self.start: int | None = None

    def update(self, motion: float, index: int) -> tuple[int, int] | None:
        if self.rest is None:
            self.rest = max(motion, self.floor)

        if not self.in_sign:
            # Learn the rest level only while resting — the old trigger's
            # baseline included signing motion and so drifted upward
            self.rest = 0.9 * self.rest + 0.1 * motion
            if motion > max(self.floor, self.rest * self.enter):
                self.in_sign, self.start, self.quiet, self.peak = True, index, 0, motion
            return None

        self.peak = max(self.peak, motion)
        exit_t = max(self.floor * 0.7, self.rest * self.exit_)
        if motion < exit_t:
            self.quiet += 1
        else:
            self.quiet = 0

        too_long = index - self.start >= self.max_len
        if self.quiet >= self.quiet_frames or too_long:
            self.in_sign = False
            end = index - (self.quiet if not too_long else 0)
            return (self.start, end) if end - self.start >= self.min_len else None
        return None

    def reset(self) -> None:
        self.in_sign, self.quiet, self.start, self.peak = False, 0, None, 0.0


def window_for(frames: list, start: int, end: int) -> np.ndarray:
    """Pad a detected segment out to the proportions of a training clip, where
    the sign sat 18% in and filled ~45% of the window."""
    total = max(len(frames) // 100, int(round((end - start) / SIGN_ACTIVE_FRAC)))
    lead = int(round(total * SIGN_ONSET_FRAC))
    lo = max(0, start - lead)
    return np.array(frames[lo:min(len(frames), lo + total)], dtype=np.float32)
