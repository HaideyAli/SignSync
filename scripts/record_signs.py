import json
import sys
import time
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from extract_landmarks import landmarks_from_frame, ensure_model

LANDMARKS_DIR  = Path("data/landmarks")
RAW_VIDEOS_DIR = Path("data/raw/videos")
MODELS_DIR     = Path("data/models")
LABELS_PATH    = Path("data/labels_50.json")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
)
MIN_TAKES   = 10
RECORD_SECS = 4
W, H = 640, 480


def personal_count(word: str) -> int:
    return len(list(LANDMARKS_DIR.glob(f"{word}_personal_*.npy")))


def next_take(word: str) -> int:
    files = list(LANDMARKS_DIR.glob(f"{word}_personal_*.npy"))
    return max((int(f.stem.rsplit("_", 1)[-1]) for f in files), default=-1) + 1


def find_ref(word: str):
    # Match WLASL files: {word}_{numeric_id}.npy — excludes personal_ files
    for npy in LANDMARKS_DIR.glob(f"{word}_[0-9]*.npy"):
        vid = RAW_VIDEOS_DIR / f"{npy.stem.rsplit('_', 1)[-1]}.mp4"
        if vid.exists():
            return cv2.VideoCapture(str(vid))
    return None


def txt(img, text, x, y, scale=0.6, color=(255, 255, 255)):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def main():
    LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "holistic_landmarker.task"
    ensure_model(MODEL_URL, model_path)

    with open(LABELS_PATH) as f:
        label_map = json.load(f)
    words = [w for w in sorted(label_map) if personal_count(w) < MIN_TAKES]
    print(f"{len(words)} words need recordings (target: {MIN_TAKES} takes each).")

    opts = mp_vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("ERROR: cannot open webcam")

    with mp_vision.HolisticLandmarker.create_from_options(opts) as detector:
        for wi, word in enumerate(words):
            ref       = find_ref(word)
            take      = next_take(word)
            buf: list = []
            rec_start = None

            while personal_count(word) < MIN_TAKES:
                ret, frame = cap.read()
                if not ret:
                    break

                # Webcam on left (mirror-flipped for natural viewing)
                display = cv2.flip(cv2.resize(frame, (W, H)), 1)
                canvas  = np.zeros((H, W * 2, 3), np.uint8)
                canvas[:, :W] = display

                # Reference on right: loop WLASL video or show lookup hint
                if ref:
                    ok, rframe = ref.read()
                    if not ok:
                        ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        _, rframe = ref.read()
                    if rframe is not None:
                        canvas[:, W:] = cv2.resize(rframe, (W, H))
                else:
                    url = f"lifeprint.com/asl101/pages-signs/{word[0]}/{word}.htm"
                    txt(canvas, f"SIGN: {word.upper()}", W + 20, H // 2 - 50, scale=1.0, color=(0, 220, 255))
                    txt(canvas, "No reference video found.", W + 20, H // 2 + 10, scale=0.5)
                    txt(canvas, "Look up at:", W + 20, H // 2 + 45, scale=0.5)
                    txt(canvas, url, W + 20, H // 2 + 75, scale=0.42, color=(150, 180, 255))

                # HUD
                hud = f"Word {wi+1}/{len(words)}   Take {take+1}/{MIN_TAKES}"
                txt(canvas, hud, 10, 28, scale=0.65, color=(0, 255, 180))
                # Large word label centred on webcam side
                (tw, _), _ = cv2.getTextSize(word.upper(), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
                txt(canvas, word.upper(), (W - tw) // 2, H - 50, scale=1.6, color=(0, 220, 255))
                txt(canvas, "SPACE=record  N=next word  Q=quit", 10, H - 12, scale=0.45, color=(160, 160, 160))

                # Recording indicator + frame capture
                if rec_start is not None:
                    elapsed = time.time() - rec_start
                    left    = max(0.0, RECORD_SECS - elapsed)
                    txt(canvas, f"REC  {left:.1f}s", 10, 68, scale=0.9, color=(30, 30, 255))

                    raw = cv2.resize(frame, (W, H))
                    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                    buf.append(landmarks_from_frame(
                        detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                    ))

                    if elapsed >= RECORD_SECS:
                        out = LANDMARKS_DIR / f"{word}_personal_{take:03d}.npy"
                        np.save(out, np.array(buf, dtype=np.float32))
                        txt(canvas, f"Saved  take {take+1}/{MIN_TAKES}", 10, 108, color=(0, 255, 0))
                        cv2.imshow("SignBridge — Record", canvas)
                        cv2.waitKey(600)   # brief pause so user sees confirmation
                        take     += 1
                        buf       = []
                        rec_start = None

                cv2.imshow("SignBridge — Record", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    if ref:
                        ref.release()
                    cv2.destroyAllWindows()
                    return
                if key == ord('n'):
                    break
                if key == ord(' ') and rec_start is None:
                    buf       = []
                    rec_start = time.time()

            if ref:
                ref.release()

    cap.release()
    cv2.destroyAllWindows()
    n = len(list(LANDMARKS_DIR.glob("*_personal_*.npy")))
    print(f"\nDone. {n} personal recordings saved to data/landmarks/")
    print("Next steps:")
    print("  cd data && zip -j landmarks.zip landmarks/*.npy")
    print("  Upload landmarks.zip to Drive and re-run train_colab.ipynb")


if __name__ == "__main__":
    main()