"""
Recounts .npy files per class and rebuilds labels_50.json with the top 50 classes.
Run after adding new landmark files from ASL-Citizens.

Usage: python scripts/rebuild_labels.py [--landmarks data/landmarks] [--out data/labels_50.json] [--n 50]
"""
import json
import argparse
from pathlib import Path
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", default="data/landmarks")
    parser.add_argument("--out", default="data/labels_50.json")
    parser.add_argument("--n", type=int, default=50, help="Number of top classes to keep")
    args = parser.parse_args()

    landmarks_dir = Path(args.landmarks)
    counts = Counter()
    for f in landmarks_dir.glob("*.npy"):
        word = "_".join(f.stem.split("_")[:-1])
        counts[word] += 1

    if not counts:
        print(f"No .npy files found in {landmarks_dir}")
        return

    top_n = [word for word, _ in counts.most_common(args.n)]
    label_map = {word: idx for idx, word in enumerate(top_n)}

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print(f"Saved {len(label_map)}-class label map -> {out_path}")
    print(f"\n{'Class':<25} {'Idx':>4} {'Samples':>8}")
    print("-" * 40)
    for word, idx in label_map.items():
        print(f"{word:<25} {idx:>4} {counts[word]:>8}")

    total = sum(counts[w] for w in top_n)
    print(f"\nTotal samples across top {args.n} classes: {total}")


if __name__ == "__main__":
    main()
