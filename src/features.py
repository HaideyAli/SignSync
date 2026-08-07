"""Landmark -> model-input preprocessing.

Split out of dataset.py to respect the 150-line rule, and because live
inference needs exactly this pipeline: the frames coming off the webcam must
be transformed identically to the training data or the model sees a different
distribution than it was trained on.

Order matters: normalise -> drop_legs -> add_presence_flags -> resample
-> compute_velocity.
"""
import numpy as np

SEQ_LEN    = 30
NUM_VALUES = 258   # raw landmark values per frame as stored on disk

# Pose landmarks 25-32 are knees/ankles/heels/feet. Measured mean MediaPipe
# visibility 0.045 — effectively never detected — yet they occupy 32 of the
# 258 values in every frame. Dropping them removes 12% pure noise.
_LEG_START = 126 + 25 * 4   # 226
KEPT_VALUES = 226           # 63 + 63 + 25 pose landmarks * 4
FLAG_VALUES = KEPT_VALUES + 2          # + left/right hand presence
OUT_VALUES  = FLAG_VALUES * 2          # 456, after appending velocity

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

    # Undetected blocks must survive as zeros. Centring/scaling an all-zero
    # hand turns it into -centre/scale (magnitude ~1.9), which reads as "hand
    # at a definite position" rather than "no hand" — and hands are missing in
    # 40-80% of frames, so this corrupted most of the data.
    l_missing = np.abs(seq[:, 0:63]).sum(1)    < 1e-6
    r_missing = np.abs(seq[:, 63:126]).sum(1)  < 1e-6

    # x,y,z only — the pose visibility channel is a probability, leave it be
    result = seq.copy()
    for i in range(0, 126, 3):
        result[:, i:i+3] = (result[:, i:i+3] - centre) / scale
    for i in range(126, 258, 4):
        result[:, i:i+3] = (result[:, i:i+3] - centre) / scale

    result[l_missing, 0:63]   = 0.0
    result[r_missing, 63:126] = 0.0
    return result


def drop_legs(seq: np.ndarray) -> np.ndarray:
    """Remove pose landmarks 25-32 (legs/feet), which are near-never detected."""
    return seq[:, :_LEG_START]


def add_presence_flags(seq: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Append explicit left/right hand-present channels.

    With zeros preserved the model *could* infer presence from an all-zero
    block, but saying so directly is cheaper to learn than detecting the
    absence of signal across 63 dimensions. Flags come from the raw landmarks
    so normalisation cannot affect them."""
    left  = (np.abs(raw[:, 0:63]).sum(1)   > 1e-6).astype(np.float32)
    right = (np.abs(raw[:, 63:126]).sum(1) > 1e-6).astype(np.float32)
    return np.concatenate([seq, left[:, None], right[:, None]], axis=1)


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
    the resampled timebase. Output: (T, OUT_VALUES)."""
    delta = np.zeros_like(seq)
    delta[1:] = seq[1:] - seq[:-1]
    return np.concatenate([seq, delta], axis=1).astype(np.float32)


def preprocess(raw: np.ndarray, seq_len: int = SEQ_LEN, augment_fn=None) -> np.ndarray:
    """Raw (T, 258) landmarks -> (seq_len, OUT_VALUES) model input.

    The single definition of the pipeline. Training and live inference both
    call this, so the webcam cannot end up seeing a different transform than
    the model was fitted on — the failure mode that silently destroys accuracy
    at deployment."""
    seq = normalise_landmarks(raw)
    if augment_fn is not None:
        seq = augment_fn(seq)                 # train only; flags come from raw
    seq = drop_legs(seq)
    seq = add_presence_flags(seq, raw)
    seq = resample(seq, seq_len)
    return compute_velocity(seq)
