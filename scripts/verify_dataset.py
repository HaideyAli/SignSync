"""
Sanity check for Phase 1.
Run from the project root:  python scripts/verify_dataset.py
Expects data/landmarks/ and data/labels.json to exist (run extract_landmarks.py first).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import ASLDataset, create_dataloaders


# Loads the dataset, prints 5 sample shapes and labels, then pulls a batch and checks its dimensions
def main():
    print("Loading dataset...")
    dataset = ASLDataset()
    print(f"  Total samples : {len(dataset)}")
    print(f"  Total classes : {len(dataset.label_map)}")

    if len(dataset) == 0:
        print("\nERROR: No samples found.")
        print("Make sure extract_landmarks.py has been run and")
        print("data/landmarks/ contains .npy files.")
        sys.exit(1)

    print("\n--- 5 sample checks ---")
    idx_to_word = {v: k for k, v in dataset.label_map.items()}
    for i in range(min(5, len(dataset))):
        seq, label = dataset[i]
        print(f"  [{i}]  shape={tuple(seq.shape)}  "
              f"label={label:3d}  word={idx_to_word[label]}")

    print("\n--- DataLoader check ---")
    train_loader, val_loader, _ = create_dataloaders(batch_size=16)
    batch_seq, batch_labels = next(iter(train_loader))
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Batch shape   : {tuple(batch_seq.shape)}  (expect [16, 30, 258])")
    print(f"  Labels shape  : {tuple(batch_labels.shape)}")

    assert batch_seq.shape == (16, 30, 258), "Unexpected batch shape!"
    print("\nSanity check PASSED.")


if __name__ == "__main__":
    main()
