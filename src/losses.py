"""Training-time objective helpers, split from train.py for the 150-line rule."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    # Down-weights easy examples so training focuses on hard/rare ones
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none",
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def mixup(seqs, labels, alpha: float):
    """Convex-combine pairs of sequences, returning the partner labels and lam.

    Regularises by a different mechanism to augmentation: rather than showing
    more variants of one sample, it forces predictions to vary linearly
    between samples, which discourages memorising individual clips.

    Note the training accuracy reported on mixed batches is not comparable to
    clean accuracy — judge the train/val gap from an eval pass instead.
    """
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(seqs.size(0), device=seqs.device)
    return lam * seqs + (1 - lam) * seqs[idx], labels[idx], lam
