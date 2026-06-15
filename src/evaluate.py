import argparse
import torch
import numpy as np
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import create_dataloaders
from model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Runs the model over the full val set and collects predictions and true labels
def get_predictions(model, loader):
    model.eval()
    all_preds, all_labels, all_logits = [], [], []

    with torch.no_grad():
        for seqs, labels in loader:
            seqs = seqs.to(DEVICE)
            logits = model(seqs)
            all_logits.append(logits.cpu())
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(labels)

    return (torch.cat(all_logits),
            torch.cat(all_preds),
            torch.cat(all_labels))


# Computes top-k accuracy from logits and true labels
def topk_accuracy(logits, labels, k):
    topk = logits.topk(k, dim=1).indices
    correct = topk.eq(labels.unsqueeze(1).expand_as(topk)).any(dim=1)
    return correct.float().mean().item()


# Saves a confusion matrix image to checkpoints/confusion_matrix.png
def save_confusion_matrix(preds, labels, idx_to_word, out_path):
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    except ImportError:
        print("sklearn/matplotlib not available — skipping confusion matrix")
        return

    cm = confusion_matrix(labels.numpy(), preds.numpy())
    fig, ax = plt.subplots(figsize=(20, 20))
    disp = ConfusionMatrixDisplay(cm, display_labels=[idx_to_word[i] for i in range(len(idx_to_word))])
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    print(f"Confusion matrix saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", default="transformer", choices=["lstm", "transformer"])
    args = parser.parse_args()

    _, val_loader, label_map = create_dataloaders()
    idx_to_word = {v: k for k, v in label_map.items()}

    model = build_model(args.arch).to(DEVICE)
    ckpt  = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    logits, preds, labels = get_predictions(model, val_loader)

    top1 = topk_accuracy(logits, labels, k=1)
    top5 = topk_accuracy(logits, labels, k=5)
    print(f"\nTop-1 accuracy : {top1*100:.1f}%")
    print(f"Top-5 accuracy : {top5*100:.1f}%")

    print("\nPer-class accuracy:")
    for idx in range(len(label_map)):
        mask    = labels == idx
        if mask.sum() == 0:
            continue
        acc     = (preds[mask] == idx).float().mean().item()
        word    = idx_to_word[idx]
        print(f"  {word:<20} {acc*100:5.1f}%  ({mask.sum().item()} samples)")

    out_path = Path(args.checkpoint).parent / "confusion_matrix.png"
    save_confusion_matrix(preds, labels, idx_to_word, out_path)


if __name__ == "__main__":
    main()
