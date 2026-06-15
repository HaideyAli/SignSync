import numpy as np

# Landmark layout in each 258-value row:
#   0:63   = left hand  (21 landmarks × x,y,z)
#   63:126 = right hand (21 landmarks × x,y,z)
#   126:258 = pose      (33 landmarks × x,y,z,visibility)

HAND_END  = 126
POSE_END  = 258


# Adds small random noise to all spatial coordinates — simulates natural signing variation
def add_noise(seq: np.ndarray, std: float = 0.01) -> np.ndarray:
    noise = np.random.normal(0, std, seq.shape).astype(np.float32)
    # Don't perturb pose visibility values (every 4th value in the pose block)
    vis_indices = np.arange(HAND_END + 3, POSE_END, 4)
    noise[:, vis_indices] = 0.0
    return seq + noise


# Shifts the sequence forward or backward in time with zero-padding at the gap
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


# Swaps left and right hands and flips x-coordinates — simulates a mirrored signer
def mirror_hands(seq: np.ndarray) -> np.ndarray:
    result = seq.copy()
    left  = seq[:, :63].copy()
    right = seq[:, 63:126].copy()

    # Flip x-coordinate (every 3rd value starting at 0) so the spatial position mirrors correctly
    left_x  = left[:,  0::3].copy()
    right_x = right[:, 0::3].copy()

    left[:,  0::3] = 1.0 - right_x
    right[:, 0::3] = 1.0 - left_x

    result[:, :63]   = left
    result[:, 63:126] = right
    return result


# Applies noise and temporal shift — mirror removed as it creates invalid ASL signs
def augment_sequence(seq: np.ndarray) -> np.ndarray:
    if np.random.random() < 0.5:
        seq = add_noise(seq, std=0.005)
    if np.random.random() < 0.5:
        seq = temporal_shift(seq)
    return seq
