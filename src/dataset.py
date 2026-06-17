import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from pathlib import Path

SEQ_LEN    = 30
NUM_VALUES = 258   # raw landmark values per frame
OUT_VALUES = 516   # after appending velocity (258 positions + 258 deltas)

# Pose landmark indices for left hip (23) and right hip (24) within the pose block
# Pose block starts at 126; each landmark is 4 values (x,y,z,vis)
_LEFT_HIP  = 126 + 23 * 4   # 218
_RIGHT_HIP = 126 + 24 * 4   # 222


def normalise_landmarks(seq: np.ndarray) -> np.ndarray:
    """Translate every frame so the torso centre (mean of hips) is at origin.
    Skips frames where both hips are missing (all zeros = undetected)."""
    left_hip  = seq[:, _LEFT_HIP  : _LEFT_HIP  + 3].copy()   # (T, 3)
    right_hip = seq[:, _RIGHT_HIP : _RIGHT_HIP + 3].copy()
    centre    = (left_hip + right_hip) / 2.0                  # (T, 3)

    # Mask frames where hips were not detected (both near zero)
    detected = (np.abs(left_hip).sum(axis=1) + np.abs(right_hip).sum(axis=1)) > 1e-6
    centre[~detected] = 0.0   # no-op for those frames

    result = seq.copy()
    for i in range(0, 126, 3):
        result[:, i:i+3] -= centre
    for i in range(126, 258, 4):
        result[:, i:i+3] -= centre

    return result


def compute_velocity(seq: np.ndarray) -> np.ndarray:
    """Append frame-to-frame deltas. Must be called BEFORE zero-padding
    so padded frames don't create spurious velocity spikes. Output: (T, 516)."""
    delta = np.zeros_like(seq)
    delta[1:] = seq[1:] - seq[:-1]
    return np.concatenate([seq, delta], axis=1).astype(np.float32)


class ASLDataset(Dataset):
    # Scans the landmarks folder and builds the list of (file, label) pairs
    def __init__(self,
                 landmarks_dir: str = "data/landmarks",
                 labels_path: str = "data/labels.json",
                 seq_len: int = SEQ_LEN,
                 augment: bool = False):
        self.seq_len = seq_len
        self.augment = augment
        self.landmarks_dir = Path(landmarks_dir)

        with open(labels_path) as f:
            self.label_map = json.load(f)

        self.samples: list[tuple[Path, int]] = []
        for npy_file in sorted(self.landmarks_dir.glob("*.npy")):
            word = "_".join(npy_file.stem.split("_")[:-1])
            if word in self.label_map:
                self.samples.append((npy_file, self.label_map[word]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        seq = np.load(path).astype(np.float32)   # (T, 258)
        seq = normalise_landmarks(seq)            # centre on torso (before padding)
        if self.augment:
            from augment import augment_sequence
            seq = augment_sequence(seq)
        seq = compute_velocity(seq)               # (T, 516) — before padding so no spike at boundary
        seq = self._pad_or_trim(seq)              # (30, 516)
        return torch.from_numpy(seq), label

    def _pad_or_trim(self, seq: np.ndarray) -> np.ndarray:
        T = seq.shape[0]
        if T >= self.seq_len:
            return seq[: self.seq_len]
        pad = np.zeros((self.seq_len - T, seq.shape[1]), dtype=np.float32)
        return np.vstack([seq, pad])


# Splits into train/val, applies weighted sampler to handle class imbalance
def create_dataloaders(
    landmarks_dir: str = "data/landmarks",
    labels_path: str = "data/labels.json",
    batch_size: int = 32,
    val_split: float = 0.15,
    seed: int = 42,
    augment: bool = False,
) -> tuple[DataLoader, DataLoader, dict]:
    base  = ASLDataset(landmarks_dir, labels_path)
    n     = len(base)
    val_n = int(n * val_split)
    perm  = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    train_idx, val_idx = perm[val_n:], perm[:val_n]

    train_ds = Subset(ASLDataset(landmarks_dir, labels_path, augment=augment), train_idx)
    val_ds   = Subset(ASLDataset(landmarks_dir, labels_path, augment=False),   val_idx)

    # Weighted sampler — rare classes get same expected frequency as common ones
    label_counts: dict[int, int] = {}
    for idx in train_idx:
        _, lbl = base.samples[idx]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    weights = [1.0 / label_counts[base.samples[i][1]] for i in train_idx]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_idx), replacement=True)

    # num_workers=0 required on Windows; sampler replaces shuffle
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,   num_workers=0)

    return train_loader, val_loader, base.label_map
