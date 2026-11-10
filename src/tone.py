"""Adaptive tone mapping for the output picture.

Replaces the fixed gamma curve that camera.py's DEFAULT_GAMMA drove. A fixed
curve cannot win: gamma < 1 brightens midtones but lifts the black point with
them, so the picture goes milky (measured: true sensor black sat at 17, gamma
0.80 raised it to 48 — grey, not black), while gamma near 1 leaves a dark room
dark. Every value is wrong in some lighting, which is why the brightness flags
were retuned four times in five commits.

Measuring the frame's own black and white points and stretching that range to
full scale fixes both directions at once: a washed-out frame gets its blacks
pushed back down, a dark frame gets its compressed histogram pulled up. Gamma
then shapes the midtones *within* the recovered range.

Costs nothing per frame: the whole curve bakes into the same 256-entry table
cv2.LUT already applied. Only the periodic re-measure costs anything, and that
runs on a 320x180 thumbnail every MEASURE_INTERVAL frames.

Kept out of camera.py so that file stays under the 150-line limit.
"""
import cv2
import numpy as np

MEASURE_INTERVAL = 15     # frames between re-measures; ~0.5s at 30fps
SMOOTHING        = 0.2    # EMA factor on the black/white points
MIN_SPAN         = 32.0   # never stretch a nearly-flat scene to full range —
                          # a blank wall would otherwise be amplified into noise
THUMB            = (320, 180)
TARGET_MID       = 0.47   # where the average pixel should sit, 0..1. Stretching
                          # alone does not reach it: a dim room can already span
                          # black to white and still average 82/255, which is
                          # the dark picture that started all this.
GAMMA_RANGE      = (0.45, 1.15)
_MID_RANGE       = (0.02, 0.90)   # keeps log() away from 0 and 1


def tone_lut(black: float, white: float, gamma: float) -> np.ndarray:
    """Map [black, white] onto [0, 255], then apply gamma to the midtones."""
    span = max(white - black, MIN_SPAN)
    x = (np.arange(256, dtype=np.float32) - black) / span
    return (np.clip(x, 0.0, 1.0) ** gamma * 255.0).astype(np.uint8)


class ToneMapper:
    """Stateful tone curve that re-measures the picture as the light changes.

    The black/white points are exponentially smoothed rather than taken raw:
    a hand crossing the frame moves the percentiles, and rebuilding the curve
    straight from each measurement makes the whole image pump in brightness.
    """

    def __init__(self, gamma: float | None = None, interval: int = MEASURE_INTERVAL):
        # gamma None -> chosen automatically each measure; a number pins it;
        # 0 turns tone mapping off entirely.
        self.gamma = gamma
        self.interval = interval
        self._black: float | None = None
        self._white: float | None = None
        self._auto_gamma: float | None = None
        self._lut: np.ndarray | None = None
        self._lut_gamma: float | None = None
        self._i = 0

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Tone-map a frame. Returns it untouched when gamma is None/0.

        Cosmetic only — never call this on what the detector sees: the model
        should keep receiving pixels resembling its training data.
        """
        if self.gamma == 0:
            return frame
        # Rebuild on schedule, on the first frame, or when gamma is changed
        # live (scripts/tune_camera.py drives it from a slider).
        if self._lut is None or self._i % self.interval == 0 or self._lut_gamma != self.gamma:
            self._remeasure(frame)
        self._i += 1
        return cv2.LUT(frame, self._lut)

    def _remeasure(self, frame: np.ndarray) -> None:
        small = cv2.resize(frame, THUMB)
        black = float(np.percentile(small, 1))
        white = float(np.percentile(small, 99))
        if self._black is None:
            self._black, self._white = black, white
        else:
            self._black += SMOOTHING * (black - self._black)
            self._white += SMOOTHING * (white - self._white)
        self._lut = tone_lut(self._black, self._white, self._pick_gamma(small))
        self._lut_gamma = self.gamma

    def _pick_gamma(self, small: np.ndarray) -> float:
        """Gamma that lands the average pixel on TARGET_MID, or the pinned one.

        Solves mid**gamma == TARGET_MID for gamma, on the already-stretched
        values — so it corrects exposure rather than fighting the stretch.
        Smoothed like the levels, or the picture breathes as the scene moves.
        """
        if self.gamma is not None:
            return self.gamma
        span = max(self._white - self._black, MIN_SPAN)
        norm = np.clip((small.astype(np.float32) - self._black) / span, 0.0, 1.0)
        mid = float(np.clip(norm.mean(), *_MID_RANGE))
        target = float(np.clip(np.log(TARGET_MID) / np.log(mid), *GAMMA_RANGE))
        self._auto_gamma = (target if self._auto_gamma is None
                            else self._auto_gamma + SMOOTHING * (target - self._auto_gamma))
        return self._auto_gamma

    @property
    def levels(self) -> tuple[float, float]:
        """Current smoothed (black, white) points — for on-screen readouts."""
        return (self._black or 0.0, self._white or 255.0)

    @property
    def effective_gamma(self) -> float:
        """The gamma actually in the current curve, auto or pinned."""
        return self.gamma if self.gamma is not None else (self._auto_gamma or 1.0)
