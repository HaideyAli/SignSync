import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import create_dataloaders
from model import build_model

CHECKPOINT_DIR = Path("checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for seqs, labels in loader:
            seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
            logits = model(seqs)
            loss   = criterion(logits, labels)
            if training:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
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
    parser.add_argument("--debug",   action="store_true")
    args = parser.parse_args()

    if args.debug:
        args.epochs = 2

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "best_model.pth"

    train_loader, val_loader, label_map = create_dataloaders(
        labels_path=args.labels, batch_size=args.batch, augment=True
    )

    model     = build_model(args.arch, num_classes=args.num_classes).to(DEVICE)
    criterion = FocalLoss(gamma=1.0, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Training {args.arch.upper()} | {DEVICE} | "
          f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val")

    if not args.debug:
        wandb.init(project="SignBridge", name=f"{args.arch}_{args.num_classes}class_v2",
                   config=vars(args))

    best_acc, no_improve = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
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
