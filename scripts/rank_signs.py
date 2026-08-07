"""Ranks the 50-word vocabulary by live-recognition accuracy against the
user's own personal recordings, so demo sentences can be built from words
that actually recognize well. Read-only diagnostic — no training, no camera.

    python scripts/rank_signs.py --checkpoint checkpoints/best_model_transformer_50_v10.pth
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dataset import personal_take, word_from_stem  # reuse the one filename parser
from predictor import SignPredictor

LANDMARKS_DIR = Path("data/landmarks")


def personal_files_by_word(directory: Path = LANDMARKS_DIR) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for f in directory.glob("*_personal_*.npy"):
        if personal_take(f.stem) is not None:
            grouped[word_from_stem(f.stem)].append(f)
    return grouped


def rank(checkpoint: str) -> list[tuple[str, float, float, int]]:
    predictor = SignPredictor(checkpoint)
    rows = []
    for word, files in sorted(personal_files_by_word().items()):
        correct, conf_sum = 0, 0.0
        for f in files:
            raw = np.load(f).astype(np.float32)
            top_word, top_conf = predictor.predict(raw)[0]
            correct += int(top_word == word)
            conf_sum += top_conf
        n = len(files)
        rows.append((word, correct / n, conf_sum / n, n))
    return sorted(rows, key=lambda r: (-r[1], -r[2]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_model_transformer_50_v10.pth")
    args = p.parse_args()

    rows = rank(args.checkpoint)
    print(f"{'word':<15}{'top-1 acc':>10}{'avg conf':>10}{'n':>5}")
    print("-" * 40)
    for word, acc, conf, n in rows:
        print(f"{word:<15}{acc*100:>9.0f}%{conf*100:>9.0f}%{n:>5}")

    weak = [w for w, acc, *_ in rows if acc < 1.0]
    print(f"\n{len(rows)} words checked. {len(rows)-len(weak)} at 100% accuracy.")
    if weak:
        print(f"Below 100%: {', '.join(weak)}")


if __name__ == "__main__":
    main()
