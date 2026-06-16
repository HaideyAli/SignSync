"""
Checks which of our 50 ASL classes exist in the ASL-Citizens dataset.
Run locally before downloading: python scripts/align_labels.py

Requires: pip install datasets
Output: prints matched/unmatched labels and saves scripts/asl_citizens_matches.json
"""
import json
from pathlib import Path


def load_our_labels(path: str = "data/labels_50.json") -> set:
    with open(path, encoding="utf-8") as f:
        return set(json.load(f).keys())


def load_asl_citizens_labels() -> set:
    from datasets import load_dataset
    print("Loading ASL-Citizens label list (metadata only, no videos)...")
    # streaming=True avoids downloading all videos
    ds = load_dataset("drdanjwalker/asl-citizens", split="train", streaming=True)
    labels = set()
    for sample in ds:
        labels.add(sample["label"].strip().lower())
        if len(labels) % 100 == 0:
            print(f"  Found {len(labels)} unique labels so far...", end="\r")
    print(f"\nASL-Citizens has {len(labels)} unique labels total.")
    return labels


def main():
    our_labels = load_our_labels()
    print(f"Our labels (50): {sorted(our_labels)}\n")

    asl_labels = load_asl_citizens_labels()

    # Normalise both to lowercase for matching
    our_norm = {w.lower(): w for w in our_labels}
    matched = {our_norm[w]: w for w in asl_labels if w in our_norm}
    unmatched = [w for w in our_labels if w.lower() not in matched]

    print(f"\nMatched: {len(matched)}/50 classes")
    print(f"Unmatched: {unmatched}")

    out = {"matched": list(matched.keys()), "unmatched": unmatched}
    out_path = Path("scripts/asl_citizens_matches.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved match results -> {out_path}")


if __name__ == "__main__":
    main()
