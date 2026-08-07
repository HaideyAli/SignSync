"""Overfitting ablations with signer-disjoint cross-validation.

Judges configurations on validation accuracy *and* the train/val gap. Gap
alone is a bad target — a model that underfits to 40/40 has no gap and is
useless — so the report shows both and the pick should be the highest val
accuracy among configurations whose gap is acceptably small.

Usage (from repo root):
    python src/sweep.py --folds 3 --epochs 60
"""
import argparse
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dataset import create_dataloaders
from losses import FocalLoss
from model import build_model
from train import run_epoch, DEVICE

# name -> build_model / optimiser / mixup overrides
CONFIGS: dict[str, dict] = {
    "baseline_d128_L2":     dict(d_model=128, layers=2, dropout=0.4, wd=1e-2, mixup=0.0),
    "smaller_d64_L2":       dict(d_model=64,  layers=2, dropout=0.4, wd=1e-2, mixup=0.0),
    "smallest_d64_L1":      dict(d_model=64,  layers=1, dropout=0.4, wd=1e-2, mixup=0.0),
    "dropout0.6":           dict(d_model=128, layers=2, dropout=0.6, wd=1e-2, mixup=0.0),
    "wd0.05":               dict(d_model=128, layers=2, dropout=0.4, wd=5e-2, mixup=0.0),
    "mixup0.4":             dict(d_model=128, layers=2, dropout=0.4, wd=1e-2, mixup=0.4),
    "small+mixup":          dict(d_model=64,  layers=2, dropout=0.5, wd=3e-2, mixup=0.4),
}


def run_one(cfg: dict, fold: int, n_folds: int, epochs: int, labels: str,
            num_classes: int, lr: float, batch: int) -> tuple[float, float]:
    """Train one configuration on one fold. Returns (best_val, train_at_best)."""
    train_loader, val_loader, _ = create_dataloaders(
        labels_path=labels, batch_size=batch, augment=True,
        n_folds=n_folds, fold=fold)

    model = build_model("transformer", num_classes=num_classes,
                        d_model=cfg["d_model"], num_layers=cfg["layers"],
                        dropout=cfg["dropout"]).to(DEVICE)
    criterion = FocalLoss(gamma=1.0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val, train_at_best, no_improve = 0.0, 0.0, 0
    for _ in range(epochs):
        run_epoch(model, train_loader, criterion, optimizer, mixup_alpha=cfg["mixup"])
        # Clean pass over train: mixed-batch accuracy is not comparable to val
        _, train_acc = run_epoch(model, train_loader, criterion)
        _, val_acc   = run_epoch(model, val_loader,   criterion)
        scheduler.step()
        if val_acc > best_val:
            best_val, train_at_best, no_improve = val_acc, train_acc, 0
        else:
            no_improve += 1
            if no_improve >= 15:
                break
    return best_val, train_at_best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds",  type=int, default=3, help="signer-disjoint CV folds")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--labels", default="data/labels_50.json")
    p.add_argument("--num_classes", type=int, default=50)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--batch",  type=int, default=32)
    p.add_argument("--only",   default="", help="comma-separated subset of config names")
    args = p.parse_args()

    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(CONFIGS)
    results: dict[str, list[tuple[float, float]]] = {}

    for name in names:
        cfg = CONFIGS[name]
        per_fold = []
        for fold in range(args.folds):
            val, tr = run_one(cfg, fold, args.folds, args.epochs, args.labels,
                              args.num_classes, args.lr, args.batch)
            per_fold.append((val, tr))
            print(f"  {name:20s} fold {fold}: val {val:.3f}  train {tr:.3f}  gap {tr-val:+.3f}")
        results[name] = per_fold

    print(f"\n{'config':22s} {'val (mean±sd)':>16s} {'train':>7s} {'gap':>7s}")
    print("-" * 56)
    rows = []
    for name, folds in results.items():
        vals = [v for v, _ in folds]
        trs  = [t for _, t in folds]
        mean = sum(vals) / len(vals)
        sd   = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
        tr   = sum(trs) / len(trs)
        rows.append((name, mean, sd, tr, tr - mean))
    for name, mean, sd, tr, gap in sorted(rows, key=lambda r: -r[1]):
        print(f"{name:22s} {mean:7.3f} ± {sd:5.3f} {tr:7.3f} {gap:+7.3f}")

    print("\nPick the highest val accuracy with an acceptable gap — minimising gap")
    print("alone rewards underfitting, which is why both columns are shown.")


if __name__ == "__main__":
    main()
