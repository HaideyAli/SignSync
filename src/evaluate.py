import argparse
import torch
import numpy as np
import json
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import create_dataloaders, personal_take
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


# Personal takes are one signer in one sitting; WLASL is many signers in varied
# conditions. A single blended accuracy hides both numbers, so report them apart.
def accuracy_by_source(preds, labels, paths) -> dict[str, tuple[float, int]]:
    hits: dict[str, list[bool]] = {}
    for pred, label, path in zip(preds.tolist(), labels.tolist(), paths):
        src = "personal" if personal_take(path.stem) is not None else "wlasl"
        hits.setdefault(src, []).append(pred == label)
    return {k: (sum(v) / len(v), len(v)) for k, v in hits.items()}


# Which sign pairs the model actually mixes up, worst first
def top_confusions(preds, labels, idx_to_word, k: int = 10):
    pairs = Counter()
    for pred, label in zip(preds.tolist(), labels.tolist()):
        if pred != label:
            pairs[(idx_to_word[label], idx_to_word[pred])] += 1
    return pairs.most_common(k)


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
    parser.add_argument("--num_classes", type=int, default=50)
    parser.add_argument("--labels", default="data/labels_50.json")
    parser.add_argument("--group_personal", action="store_true",
                        help="hold out the last personal takes entirely (leakage-reduced split)")
    args = parser.parse_args()

    _, val_loader, label_map = create_dataloaders(labels_path=args.labels,
                                                  group_personal=args.group_personal)
    idx_to_word = {v: k for k, v in label_map.items()}

    # val_loader has shuffle=False, so prediction order matches these paths
    val_subset = val_loader.dataset
    val_paths  = [val_subset.dataset.samples[i][0] for i in val_subset.indices]

    model = build_model(args.arch, num_classes=args.num_classes).to(DEVICE)
    ckpt  = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    logits, preds, labels = get_predictions(model, val_loader)

    top1 = topk_accuracy(logits, labels, k=1)
    top5 = topk_accuracy(logits, labels, k=5)
    split_name = "grouped (personal takes held out)" if args.group_personal else "random"
    print(f"\nSplit          : {split_name}")
    print(f"Top-1 accuracy : {top1*100:.1f}%")
    print(f"Top-5 accuracy : {top5*100:.1f}%")

    print("\nAccuracy by source:")
    for src, (acc, n) in sorted(accuracy_by_source(preds, labels, val_paths).items()):
        print(f"  {src:<10} {acc*100:5.1f}%  ({n} samples)")

    print("\nMost confused pairs (true -> predicted):")
    for (true_word, pred_word), count in top_confusions(preds, labels, idx_to_word):
        print(f"  {true_word:<15} -> {pred_word:<15} {count}x")

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
