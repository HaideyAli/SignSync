import argparse
import torch
import wandb
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import create_dataloaders
from losses import FocalLoss, mixup
from model import build_model

CHECKPOINT_DIR = Path("checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, optimizer=None, mixup_alpha: float = 0.0):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for seqs, labels in loader:
            seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
            if training and mixup_alpha > 0:
                seqs, labels_b, lam = mixup(seqs, labels, mixup_alpha)
                logits = model(seqs)
                loss   = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels_b)
            else:
                logits = model(seqs)
                loss   = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)
    return total_loss / total, correct / total


def maybe_save(model, val_acc, best_acc, path):
    if val_acc > best_acc:
        torch.save({"model_state": model.state_dict()}, path)
        return True, val_acc
    return False, best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",    default="transformer", choices=["lstm", "transformer"])
    parser.add_argument("--epochs",  type=int, default=150)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=50)
    parser.add_argument("--labels",  default="data/labels_50.json")
    parser.add_argument("--group_personal", action="store_true",
                        help="hold out the last personal takes entirely (leakage-reduced split)")
    parser.add_argument("--n_folds", type=int, default=0,
                        help="signer-disjoint CV folds; 0 disables (use --group_personal instead)")
    parser.add_argument("--fold",    type=int, default=0)
    # Capacity / regularisation knobs for overfitting ablations
    parser.add_argument("--d_model",  type=int,   default=128)
    parser.add_argument("--layers",   type=int,   default=2)
    parser.add_argument("--nhead",    type=int,   default=4)
    parser.add_argument("--dropout",  type=float, default=0.4)
    parser.add_argument("--wd",       type=float, default=1e-2)
    parser.add_argument("--mixup",    type=float, default=0.0,
                        help="mixup alpha; 0 disables")
    parser.add_argument("--debug",   action="store_true")
    args = parser.parse_args()

    if args.debug:
        args.epochs = 2

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "best_model.pth"

    train_loader, val_loader, label_map = create_dataloaders(
        labels_path=args.labels, batch_size=args.batch, augment=True,
        group_personal=args.group_personal, n_folds=args.n_folds, fold=args.fold
    )

    model     = build_model(args.arch, num_classes=args.num_classes,
                            d_model=args.d_model, num_layers=args.layers,
                            nhead=args.nhead, dropout=args.dropout).to(DEVICE)
    criterion = FocalLoss(gamma=1.0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Training {args.arch.upper()} | {DEVICE} | "
          f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val")

    if not args.debug:
        wandb.init(project="SignBridge", name=f"{args.arch}_{args.num_classes}class_v4",
                   config=vars(args))

    best_acc, no_improve = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer,
                                          mixup_alpha=args.mixup)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion)
        scheduler.step()

        saved, best_acc = maybe_save(model, val_acc, best_acc, ckpt_path)
        no_improve      = 0 if saved else no_improve + 1

        print(f"Epoch {epoch:3d} | train {train_acc:.3f} | val {val_acc:.3f}"
              + (" *" if saved else ""))

        if not args.debug:
            wandb.log({"train_loss": train_loss, "train_acc": train_acc,
                       "val_loss": val_loss,   "val_acc":   val_acc,
                       "best_val_acc": best_acc, "lr": scheduler.get_last_lr()[0]})

        if no_improve >= 15:
            print("Early stopping.")
            break

    print(f"Done. Best val accuracy: {best_acc:.3f} -> {ckpt_path}")
    if not args.debug:
        wandb.finish()


if __name__ == "__main__":
    main()
