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

    WLASL clips (take is None) are split randomly here, which leaves the same
    signer on both sides — see signer_fold_indices for the leakage-free version.
    """
    personal_val   = [i for i, t in enumerate(takes) if t is not None and t >= holdout_from]
    personal_train = [i for i, t in enumerate(takes) if t is not None and t <  holdout_from]
    wlasl          = [i for i, t in enumerate(takes) if t is None]

    val_n = int(len(wlasl) * val_split)
    perm  = torch.randperm(len(wlasl), generator=torch.Generator().manual_seed(seed)).tolist()
    wlasl_val   = [wlasl[i] for i in perm[:val_n]]
    wlasl_train = [wlasl[i] for i in perm[val_n:]]

    return personal_train + wlasl_train, personal_val + wlasl_val


def signer_folds(signers: list[int | None], n_folds: int, seed: int) -> list[list[int]]:
    """Partition distinct signers into n_folds groups of similar clip count.

    True leave-one-signer-out is not usable here: of 64 signers most have
    under 12 clips and none covers all 50 classes, so per-fold accuracy would
    be measured on a handful of samples over a partial label set. Grouping
    signers into a few balanced folds keeps every fold signer-disjoint while
    leaving each val set large enough to mean something."""
    counts: dict[int, int] = {}
    for s in signers:
        if s is not None:
            counts[s] = counts.get(s, 0) + 1

    # Shuffle first so equal-sized signers land differently per seed, then
    # greedy bin-pack largest-first: each signer joins the smallest fold.
    keys = list(counts)
    perm = torch.randperm(len(keys), generator=torch.Generator().manual_seed(seed)).tolist()
    order = sorted((keys[i] for i in perm), key=lambda s: -counts[s])

    folds: list[list[int]] = [[] for _ in range(n_folds)]
    sizes = [0] * n_folds
    for s in order:
        j = min(range(n_folds), key=lambda k: sizes[k])
        folds[j].append(s)
        sizes[j] += counts[s]
    return folds


def signer_fold_indices(takes: list[int | None],
                        signers: list[int | None],
                        fold_signers: list[int],
                        holdout_from: int) -> tuple[list[int], list[int]]:
    """Signer-disjoint split: every clip from fold_signers goes to val.

    Personal takes still split by take number, since they are all one person
    and cannot be made person-independent."""
    train, val = [], []
    for i, (t, s) in enumerate(zip(takes, signers)):
        if t is not None:                       # personal recording
            (val if t >= holdout_from else train).append(i)
        else:                                   # WLASL clip
            (val if s in fold_signers else train).append(i)
    return train, val
