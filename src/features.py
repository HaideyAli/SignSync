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


# MediaPipe pose left/right landmark pairs; 0 (nose) is central and has no partner
_POSE_MIRROR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14),
                      (15, 16), (17, 18), (19, 20), (21, 22), (23, 24), (25, 26),
                      (27, 28), (29, 30), (31, 32)]


def mirror_landmarks(seq: np.ndarray) -> np.ndarray:
    """Horizontally mirror a clip: swap hand blocks, swap pose left/right pairs,
    and flip x. Call on RAW landmarks, before normalisation, while x is still
    in [0,1] image space.

    Personal webcam takes are ~51% left-hand-dominant while WLASL clips are
    ~51% right-hand-dominant, so the signing hand sits in opposite blocks and
    the model would otherwise have to learn every sign twice."""
    l_missing = np.abs(seq[:, 0:63]).sum(1)    < 1e-6
    r_missing = np.abs(seq[:, 63:126]).sum(1)  < 1e-6
    p_missing = np.abs(seq[:, 126:258]).sum(1) < 1e-6

    out = seq.copy()
    out[:, 0:63]   = seq[:, 63:126]
    out[:, 63:126] = seq[:, 0:63]
    for a, b in _POSE_MIRROR_PAIRS:
        ia, ib = 126 + a * 4, 126 + b * 4
        out[:, ia:ia + 4] = seq[:, ib:ib + 4]
        out[:, ib:ib + 4] = seq[:, ia:ia + 4]

    # x only — y is unaffected by a horizontal flip, z is depth, vis is a probability
    out[:, 0:126:3]   = 1.0 - out[:, 0:126:3]
    out[:, 126:258:4] = 1.0 - out[:, 126:258:4]

    # Undetected blocks must stay zero; the flip above would have made them 1.0.
    # Masks swap sides along with the data.
    out[r_missing, 0:63]    = 0.0
    out[l_missing, 63:126]  = 0.0
    out[p_missing, 126:258] = 0.0
    return out


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
