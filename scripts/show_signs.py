"""Reference viewer: loops the WLASL clip for each of the 50 signs.

Runs standalone (no webcam, no MediaPipe), so keep it open in a second window
next to inference.py while testing.

    python scripts/show_signs.py
    python scripts/show_signs.py --word drink   # jump straight to one sign

Keys:  N / SPACE next    P previous    S slow-mo    R replay    Q quit
"""
import argparse
import json
from pathlib import Path

import cv2

LANDMARKS_DIR = Path("data/landmarks")
VIDEOS_DIR    = Path("data/raw/videos")
LABELS_PATH   = Path("data/labels_50.json")
W, H = 640, 480


def find_refs(word: str) -> list[Path]:
    """Every WLASL video we hold for this word. Personal recordings are skipped
    — they are landmarks only, no video was ever saved."""
    vids = []
    for npy in sorted(LANDMARKS_DIR.glob(f"{word}_[0-9]*.npy")):
        vid = VIDEOS_DIR / f"{npy.stem.rsplit('_', 1)[-1]}.mp4"
        if vid.exists():
            vids.append(vid)
    return vids


def txt(img, s, x, y, scale=0.7, color=(255, 255, 255)):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def show(word: str, index: int, total: int) -> str:
    """Loop this word's reference clip. Returns 'next' | 'prev' | 'quit'."""
    refs = find_refs(word)
    cap  = cv2.VideoCapture(str(refs[0])) if refs else None
    delay, ref_i = 33, 0

    while True:
        canvas = None
        if cap is not None:
            ok, frame = cap.read()
            if not ok:                      # loop the clip
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if ok:
                canvas = cv2.resize(frame, (W, H))

        if canvas is None:                  # no video on disk for this word
            import numpy as np
            canvas = np.zeros((H, W, 3), dtype="uint8")
            txt(canvas, "no reference video", 20, H // 2 - 20, 0.8, (90, 90, 220))
            txt(canvas, f"lifeprint.com/asl101/pages-signs/{word[0]}/{word}.htm",
                20, H // 2 + 20, 0.45, (150, 180, 255))

        txt(canvas, word.upper(), 20, 44, 1.3, (0, 220, 255))
        txt(canvas, f"{index + 1}/{total}"
                    + (f"   clip {ref_i + 1}/{len(refs)}" if len(refs) > 1 else "")
                    + ("   SLOW" if delay > 33 else ""), 20, 74, 0.55)
        txt(canvas, "N next   P prev   S slow   R replay clip   Q quit",
            20, H - 16, 0.5, (170, 170, 170))

        cv2.imshow("SignBridge - Reference", canvas)
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("n"), ord(" ")):
            if cap: cap.release()
            return "next"
        if key == ord("p"):
            if cap: cap.release()
            return "prev"
        if key == ord("q"):
            if cap: cap.release()
            return "quit"
        if key == ord("s"):
            delay = 100 if delay == 33 else 33
        if key == ord("r") and refs:        # cycle to the next clip of this word
            ref_i = (ref_i + 1) % len(refs)
            if cap: cap.release()
            cap = cv2.VideoCapture(str(refs[ref_i]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--word", default="", help="start at this sign")
    args = p.parse_args()

    if not LABELS_PATH.exists():
        raise SystemExit(f"{LABELS_PATH} not found — run from the repo root")
    words = sorted(json.load(open(LABELS_PATH)))

    if not VIDEOS_DIR.exists():
        print(f"warning: {VIDEOS_DIR} missing — words will show lookup links only")

    i = words.index(args.word) if args.word in words else 0
    while True:
        action = show(words[i], i, len(words))
        if action == "quit":
            break
        i = (i + 1) % len(words) if action == "next" else (i - 1) % len(words)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
