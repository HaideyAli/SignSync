import numpy as np

# Landmark layout in each 258-value row:
#   0:63   = left hand  (21 landmarks x,y,z)
#   63:126 = right hand (21 landmarks x,y,z)
#   126:258 = pose      (33 landmarks x,y,z,visibility)

HAND_END = 126
POSE_END = 258


def add_noise(seq: np.ndarray, rel_std: float = 0.02) -> np.ndarray:
    """Jitter sized relative to the sequence's own spread, not an absolute
    constant — landmarks are scale-normalised upstream, so a fixed sigma
    silently changes strength whenever that normalisation changes."""
    spread = float(seq[:, :HAND_END].std()) or 1.0
    noise = np.random.normal(0, rel_std * spread, seq.shape).astype(np.float32)
    vis_indices = np.arange(HAND_END + 3, POSE_END, 4)
    noise[:, vis_indices] = 0.0
    return seq + noise


def frame_dropout(seq: np.ndarray, max_frames: int = 3) -> np.ndarray:
    """Zero the hand landmarks in a few random frames. MediaPipe loses hand
    detections mid-sign on a live webcam, so train for it — matches the
    "missing landmarks become zeros" rule the pipeline already follows."""
    result = seq.copy()
    n = np.random.randint(1, max_frames + 1)
    for idx in np.random.choice(len(seq), size=min(n, len(seq)), replace=False):
        result[idx, :HAND_END] = 0.0
    return result


def temporal_shift(seq: np.ndarray, max_shift: int = 4) -> np.ndarray:
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return seq
    result = np.zeros_like(seq)
    if shift > 0:
        result[shift:] = seq[:-shift]
    else:
        result[:shift] = seq[-shift:]
    return result


def speed_jitter(seq: np.ndarray, low: float = 0.7, high: float = 1.35) -> np.ndarray:
    """Randomly stretch or compress the sequence in time, then resample back to T frames."""
    T, D = seq.shape
    factor = np.random.uniform(low, high)
    new_len = max(2, int(round(T * factor)))
    # Resample seq to new_len
    src = np.linspace(0, T - 1, new_len)
    tmp = np.zeros((new_len, D), dtype=np.float32)
    for i, idx in enumerate(src):
        lo = int(idx); hi = min(lo + 1, T - 1); a = idx - lo
        tmp[i] = (1 - a) * seq[lo] + a * seq[hi]
    # Resample back to T
    dst = np.linspace(0, new_len - 1, T)
    result = np.zeros((T, D), dtype=np.float32)
    for i, idx in enumerate(dst):
        lo = int(idx); hi = min(lo + 1, new_len - 1); a = idx - lo
        result[i] = (1 - a) * tmp[lo] + a * tmp[hi]
    return result


def hand_scale_jitter(seq: np.ndarray, scale_range: float = 0.15) -> np.ndarray:
    """Scale hand landmark distances from the wrist by a random factor per hand."""
    result = seq.copy()
    scale = 1.0 + np.random.uniform(-scale_range, scale_range)
    for start in (0, 63):
        wrist = seq[:, start:start + 3].copy()
        hand = result[:, start:start + 63].reshape(-1, 21, 3)
        wrist_exp = wrist[:, np.newaxis, :]
        hand = wrist_exp + scale * (hand - wrist_exp)
        result[:, start:start + 63] = hand.reshape(-1, 63)
    return result


# Superseded by features.mirror_landmarks, which also swaps the pose left/right
# pairs and preserves undetected-landmark zeros. This version did neither.


def augment_sequence(seq: np.ndarray) -> np.ndarray:
    if np.random.random() < 0.5:
        seq = add_noise(seq)
    if np.random.random() < 0.5:
        seq = temporal_shift(seq)
    if np.random.random() < 0.5:
        seq = speed_jitter(seq)
    if np.random.random() < 0.5:
        seq = hand_scale_jitter(seq)
    if np.random.random() < 0.3:
        seq = frame_dropout(seq)
    return seq
