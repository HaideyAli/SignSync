import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path

SEQ_LEN = 30
NUM_VALUES = 258


class ASLDataset(Dataset):
    """Loads .npy landmark sequences and returns (tensor, label) pairs.

    Each .npy file is expected at data/landmarks/{word}_{index}.npy
    and contains a variable-length (T, 258) float32 array.
    __getitem__ always returns exactly (SEQ_LEN, 258).
    """

    # Scans the landmarks folder and builds the list of (file, label) pairs to load from
    def __init__(self,
                 landmarks_dir: str = "data/landmarks",
                 labels_path: str = "data/labels.json",
                 seq_len: int = SEQ_LEN,
                 augment: bool = False):
        self.seq_len = seq_len
        self.augment = augment
        self.landmarks_dir = Path(landmarks_dir)

        with open(labels_path) as f:
            self.label_map = json.load(f)   # {word: int}

        self.samples: list[tuple[Path, int]] = []
        for npy_file in sorted(self.landmarks_dir.glob("*.npy")):
            word = "_".join(npy_file.stem.split("_")[:-1])
            if word in self.label_map:
                self.samples.append((npy_file, self.label_map[word]))

    # Returns the total number of video samples so PyTorch knows how big the dataset is
    def __len__(self) -> int:
        return len(self.samples)

    # Loads one .npy file by index, pads or trims it to 30 frames, and returns it as a tensor
    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        seq = np.load(path).astype(np.float32)   # (T, 258)
        seq = self._pad_or_trim(seq)              # (seq_len, 258)
        if self.augment:
            from augment import augment_sequence
            seq = augment_sequence(seq)
        return torch.from_numpy(seq), label

    # Forces every sequence to exactly 30 frames — trims if too long, adds zero rows if too short
    def _pad_or_trim(self, seq: np.ndarray) -> np.ndarray:
        T = seq.shape[0]
        if T >= self.seq_len:
            return seq[: self.seq_len]
        pad = np.zeros((self.seq_len - T, NUM_VALUES), dtype=np.float32)
        return np.vstack([seq, pad])


# Splits the dataset into train/val and wraps both in DataLoaders ready for the training loop
# Augmentation is applied to train only — val always sees clean unmodified sequences
def create_dataloaders(
    landmarks_dir: str = "data/landmarks",
    labels_path: str = "data/labels.json",
    batch_size: int = 32,
    val_split: float = 0.15,
    seed: int = 42,
    augment: bool = False,
) -> tuple[DataLoader, DataLoader, dict]:
    """Return (train_loader, val_loader, label_map)."""
    # Build index split using a seeded permutation so train/val are always the same samples
    base   = ASLDataset(landmarks_dir, labels_path)
    n      = len(base)
    val_n  = int(n * val_split)
    perm   = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    train_idx, val_idx = perm[val_n:], perm[:val_n]

    # Two separate dataset instances so augment can differ between train and val
    train_ds = Subset(ASLDataset(landmarks_dir, labels_path, augment=augment), train_idx)
    val_ds   = Subset(ASLDataset(landmarks_dir, labels_path, augment=False),   val_idx)

    # Weighted sampler so rare classes aren't drowned out by common ones
    label_counts: dict[int, int] = {}
    for idx in train_idx:
        _, lbl = base.samples[idx]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    weights = [1.0 / label_counts[base.samples[i][1]] for i in train_idx]
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(train_idx), replacement=True)

    # num_workers=0 required on Windows; sampler is mutually exclusive with shuffle
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,   num_workers=0)

    return train_loader, val_loader, base.label_map
