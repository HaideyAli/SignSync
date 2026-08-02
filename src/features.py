"""Landmark -> model-input preprocessing.

Split out of dataset.py to respect the 150-line rule, and because live
inference needs exactly this pipeline: the frames coming off the webcam must
be transformed identically to the training data or the model sees a different
distribution than it was trained on.

Order matters: normalise -> resample -> compute_velocity.
"""
import numpy as np

SEQ_LEN    = 30
NUM_VALUES = 258   # raw landmark values per frame
OUT_VALUES = 516   # after appending velocity (258 positions + 258 deltas)

# Pose block starts at 126; each landmark is 4 values (x,y,z,vis)
_LEFT_HIP       = 126 + 23 * 4   # 218
_RIGHT_HIP      = 126 + 24 * 4   # 222
_LEFT_SHOULDER  = 126 + 11 * 4   # 170
_RIGHT_SHOULDER = 126 + 12 * 4   # 174


def normalise_landmarks(seq: np.ndarray) -> np.ndarray:
    """Centre every frame on the torso (mean of hips), then divide by shoulder
    width so camera distance cancels out — otherwise the same sign looks
    different near vs far. Undetected hips/shoulders are left alone."""
    left_hip  = seq[:, _LEFT_HIP  : _LEFT_HIP  + 3].copy()   # (T, 3)
    right_hip = seq[:, _RIGHT_HIP : _RIGHT_HIP + 3].copy()
    centre    = (left_hip + right_hip) / 2.0                  # (T, 3)

    # Mask frames where hips were not detected (both near zero)
    detected = (np.abs(left_hip).sum(axis=1) + np.abs(right_hip).sum(axis=1)) > 1e-6
    centre[~detected] = 0.0   # no-op for those frames

    # Shoulder width per frame; 1.0 where shoulders are missing so the divide is a no-op
    span  = (seq[:, _LEFT_SHOULDER  : _LEFT_SHOULDER  + 3]
             - seq[:, _RIGHT_SHOULDER : _RIGHT_SHOULDER + 3])
    scale = np.linalg.norm(span, axis=1, keepdims=True)       # (T, 1)
    scale[scale < 1e-6] = 1.0

    # x,y,z only — the pose visibility channel is a probability, leave it be
    result = seq.copy()
    for i in range(0, 126, 3):
        result[:, i:i+3] = (result[:, i:i+3] - centre) / scale
    for i in range(126, 258, 4):
        result[:, i:i+3] = (result[:, i:i+3] - centre) / scale

    return result


def resample(seq: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Stretch or squeeze the whole clip to exactly seq_len frames.

    Replaces truncation: seq[:30] kept only the first 30 frames, which is 49%
    of a 61-frame clip but 26% of a 117-frame one, so how much of a sign the
    model saw depended on how fast it was signed. Interpolating covers 100%
    of every clip and removes the need for zero-padding short ones."""
    T = seq.shape[0]
    if T == seq_len:
        return seq.astype(np.float32)
    src = np.linspace(0, T - 1, seq_len)
    lo  = np.floor(src).astype(int)
    hi  = np.minimum(lo + 1, T - 1)
    a   = (src - lo)[:, None]
    return ((1 - a) * seq[lo] + a * seq[hi]).astype(np.float32)


def compute_velocity(seq: np.ndarray) -> np.ndarray:
    """Append frame-to-frame deltas. Call AFTER resampling so the deltas match
    the resampled timebase. Output: (T, 516)."""
    delta = np.zeros_like(seq)
    delta[1:] = seq[1:] - seq[:-1]
    return np.concatenate([seq, delta], axis=1).astype(np.float32)
