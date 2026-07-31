"""Train/val index selection.

Kept separate from dataset.py so that file stays under the 150-line limit.
Deliberately has no dataset.py import — it works on plain index lists so the
two modules cannot form an import cycle.
"""
import torch


def random_indices(n: int, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    """Shuffle every sample and slice off val_split as validation."""
    val_n = int(n * val_split)
    perm  = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return perm[val_n:], perm[:val_n]


def grouped_indices(takes: list[int | None],
                    val_split: float,
                    seed: int,
                    holdout_from: int) -> tuple[list[int], list[int]]:
    """Leakage-reduced split.

    All 10 takes of a personal recording came from one sitting, so they share
    lighting, clothing and framing. A random split scatters near-duplicates
    across train and val and inflates the score. Here every personal take
    numbered >= holdout_from goes to val and never appears in train.

    WLASL clips (take is None) come from many signers already, so they keep
    the plain random split.
    """
    personal_val   = [i for i, t in enumerate(takes) if t is not None and t >= holdout_from]
    personal_train = [i for i, t in enumerate(takes) if t is not None and t <  holdout_from]
    wlasl          = [i for i, t in enumerate(takes) if t is None]

    val_n = int(len(wlasl) * val_split)
    perm  = torch.randperm(len(wlasl), generator=torch.Generator().manual_seed(seed)).tolist()
    wlasl_val   = [wlasl[i] for i in perm[:val_n]]
    wlasl_train = [wlasl[i] for i in perm[val_n:]]

    return personal_train + wlasl_train, personal_val + wlasl_val
