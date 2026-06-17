import numpy as np

# Landmark layout in each 258-value row:
#   0:63   = left hand  (21 landmarks x,y,z)
#   63:126 = right hand (21 landmarks x,y,z)
#   126:258 = pose      (33 landmarks x,y,z,visibility)

HAND_END = 126
POSE_END = 258


def add_noise(seq: np.ndarray, std: float = 0.005) -> np.ndarray:
    noise = np.random.normal(0, std, seq.shape).astype(np.float32)
    vis_indices = np.arange(HAND_END + 3, POSE_END, 4)
    noise[:, vis_indices] = 0.0
    return seq + noise


def temporal_shift(seq: np.ndarray, max_shift: int = 2) -> np.ndarray:
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return seq
    result = np.zeros_like(seq)
    if shift > 0:
        result[shift:] = seq[:-shift]
    else:
        result[:shift] = seq[-shift:]
    return result


def speed_jitter(seq: np.ndarray, low: float = 0.8, high: float = 1.2) -> np.ndarray:
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


def hand_scale_jitter(seq: np.ndarray, scale_range: float = 0.1) -> np.ndarray:
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


# mirror_hands is defined but not used — swaps hands, creates invalid ASL signs
def mirror_hands(seq: np.ndarray) -> np.ndarray:
    result = seq.copy()
    left, right = seq[:, :63].copy(), seq[:, 63:126].copy()
    left_x, right_x = left[:, 0::3].copy(), right[:, 0::3].copy()
    left[:, 0::3] = 1.0 - right_x
    right[:, 0::3] = 1.0 - left_x
    result[:, :63] = left
    result[:, 63:126] = right
    return result


def augment_sequence(seq: np.ndarray) -> np.ndarray:
    if np.random.random() < 0.5:
        seq = add_noise(seq)
    if np.random.random() < 0.5:
        seq = temporal_shift(seq)
    if np.random.random() < 0.5:
        seq = speed_jitter(seq)
    if np.random.random() < 0.5:
        seq = hand_scale_jitter(seq)
    return seq
