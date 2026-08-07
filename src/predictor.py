"""Checkpoint wrapper and capture timing for live recognition.

The timing constants are not arbitrary — they mirror how the training data was
collected by scripts/record_signs.py, because the model only ever saw signs in
that geometry:

  * clips were exactly RECORD_SECS long at ~22 fps (median 88 frames)
  * measured over the personal takes, signing starts ~18% into the clip and
    ends ~63% through, leaving idle padding at both ends

A free-running sliding window breaks both, which is why short signs (cool 23%
active, no 37%, yes 39%) failed live while longer ones survived.
"""
import numpy as np
import torch

from features import preprocess
from model import build_model

RECORD_SECS     = 4.0    # must match scripts/record_signs.py
SIGN_ONSET_FRAC = 0.18   # measured mean onset position within a take
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SignPredictor:
    """Turns a buffer of raw (258,) frames into ranked predictions."""

    def __init__(self, checkpoint: str):
        ckpt = torch.load(checkpoint, map_location=DEVICE)
        self.arch        = ckpt.get("arch", "transformer")
        self.num_classes = ckpt.get("num_classes", 50)
        self.idx_to_word = {v: k for k, v in ckpt["label_map"].items()}
        self.model = build_model(self.arch, num_classes=self.num_classes).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, frames, k: int = 3) -> list[tuple[str, float]]:
        raw = np.array(frames, dtype=np.float32)          # (T, 258)
        x   = torch.from_numpy(preprocess(raw)).unsqueeze(0).to(DEVICE)
        probs = torch.softmax(self.model(x), dim=1)[0]
        top = torch.topk(probs, min(k, self.num_classes))
        return [(self.idx_to_word[int(i)], float(p))
                for p, i in zip(top.values, top.indices)]


def hands_visible(frames, recent: int = 10) -> bool:
    """True if either hand was detected recently — stops the engine predicting
    confidently over an empty frame."""
    return any(np.abs(f[:126]).sum() > 1e-6 for f in frames[-recent:])


class MotionTrigger:
    """Fires on a sustained rise in hand motion, for --auto mode.

    Idle and active frames overlap heavily in absolute motion, so a fixed
    threshold misfires. This tracks a running baseline and fires only when
    motion exceeds a multiple of it for several consecutive frames.
    """

    def __init__(self, factor: float = 3.0, sustain: int = 2, floor: float = 0.3):
        self.factor, self.sustain, self.floor = factor, sustain, floor
        self.history: list[float] = []
        self.run = 0

    def update(self, prev, cur) -> bool:
        if prev is None or np.abs(cur[:126]).sum() < 1e-6:
            self.run = 0
            return False
        motion = float(np.abs(cur[:126] - prev[:126]).sum())
        self.history.append(motion)
        if len(self.history) > 60:
            self.history.pop(0)
        baseline = float(np.median(self.history)) if len(self.history) >= 15 else self.floor
        threshold = max(self.floor, baseline * self.factor)
        self.run = self.run + 1 if motion > threshold else 0
        if self.run >= self.sustain:
            self.run = 0
            return True
        return False
