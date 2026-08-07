import json
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from pathlib import Path

from features import (SEQ_LEN, add_presence_flags, drop_legs,
                      normalise_landmarks, resample, compute_velocity)
from splits import random_indices, grouped_indices

_PERSONAL_RE = re.compile(r"^(.+)_personal_(\d+)$")


def word_from_stem(stem: str) -> str:
    """WLASL files: {word}_{numeric_id} -> word. Personal recordings:
    {word}_personal_{take} -> word (the trailing numeric-id split alone
    would otherwise leave "personal" attached to the word)."""
    m = _PERSONAL_RE.match(stem)
    return m.group(1) if m else "_".join(stem.split("_")[:-1])


def personal_take(stem: str) -> int | None:
    """Take index for a personal recording, or None for a WLASL clip."""
    m = _PERSONAL_RE.match(stem)
    return int(m.group(2)) if m else None

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
            word = word_from_stem(npy_file.stem)
            if word in self.label_map:
                self.samples.append((npy_file, self.label_map[word]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        seq = np.load(path).astype(np.float32)   # (T, 258)
        raw = seq                                 # keep pre-normalisation copy for flags
        seq = normalise_landmarks(seq)            # centre + scale, missing hands stay zero
        if self.augment:
            from augment import augment_sequence
            seq = augment_sequence(seq)
        seq = drop_legs(seq)                      # (T, 226) — legs are never detected
        seq = add_presence_flags(seq, raw)        # (T, 228)
        seq = resample(seq, self.seq_len)         # (30, 228) — whole clip, no truncation
        seq = compute_velocity(seq)               # (30, 456) — deltas on the resampled timebase
        return torch.from_numpy(seq), label


# Splits into train/val, applies weighted sampler to handle class imbalance
def create_dataloaders(
    landmarks_dir: str = "data/landmarks",
    labels_path: str = "data/labels.json",
    batch_size: int = 32,
    val_split: float = 0.15,
    seed: int = 42,
    augment: bool = False,
    group_personal: bool = False,
    holdout_from: int = 8,
) -> tuple[DataLoader, DataLoader, dict]:
    base = ASLDataset(landmarks_dir, labels_path)

    if group_personal:
        takes = [personal_take(path.stem) for path, _ in base.samples]
        train_idx, val_idx = grouped_indices(takes, val_split, seed, holdout_from)
    else:
        train_idx, val_idx = random_indices(len(base), val_split, seed)

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
