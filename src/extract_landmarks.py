import cv2
import mediapipe as mp
import numpy as np
import json
import urllib.request
from pathlib import Path
from tqdm import tqdm
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

RAW_DIR       = Path("data/raw")
LANDMARKS_DIR = Path("data/landmarks")
LABELS_PATH   = Path("data/labels.json")
MODELS_DIR    = Path("data/models")

NUM_VALUES = 258  # 21*3 left + 21*3 right + 33*4 pose

HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
)


# Downloads the holistic model .task file from Google if it isn't already on disk
def ensure_model(url: str, dest: Path):
    if not dest.exists():
        print(f"Downloading {dest.name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"Saved {dest.name}")


# Reads wlasl_class_list.txt and returns {word: idx} and {idx: word} lookup tables
def load_class_list(raw_dir: Path) -> tuple[dict, dict]:
    word_to_idx, idx_to_word = {}, {}
    with open(raw_dir / "wlasl_class_list.txt") as f:
        for line in f:
            line = line.strip()
            if line:
                idx, word = line.split("\t", 1)
                idx = int(idx)
                word_to_idx[word] = idx
                idx_to_word[idx] = word
    return word_to_idx, idx_to_word


# Reads nslt_100.json and returns {video_id: label_index} for every video in the 100-word set
def load_video_label_map(raw_dir: Path) -> dict:
    with open(raw_dir / "nslt_100.json") as f:
        data = json.load(f)
    return {vid_id: info["action"][0] for vid_id, info in data.items()}


# Converts one HolisticLandmarkerResult into a flat 258-number array; missing landmarks fill with zeros
def landmarks_from_frame(result) -> np.ndarray:
    def hand(lms):
        if lms:
            return np.array([[lm.x, lm.y, lm.z]
                             for lm in lms], dtype=np.float32).flatten()
        return np.zeros(21 * 3, dtype=np.float32)

    def pose(lms):
        if lms:
            return np.array([[lm.x, lm.y, lm.z, lm.visibility or 0.0]
                             for lm in lms], dtype=np.float32).flatten()
        return np.zeros(33 * 4, dtype=np.float32)

    return np.concatenate([
        hand(result.left_hand_landmarks),
        hand(result.right_hand_landmarks),
        pose(result.pose_landmarks),
    ])


# Opens a video file, runs MediaPipe on every frame, and returns the full sequence as a 2D array
def process_video(video_path: Path, detector) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to fixed size — prevents SegmentationSmoother from crashing on mixed-resolution videos
        frame    = cv2.resize(frame, (640, 480))
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frames.append(landmarks_from_frame(detector.detect(mp_image)))

    cap.release()
    if frames:
        return np.array(frames, dtype=np.float32)
    return np.zeros((1, NUM_VALUES), dtype=np.float32)


# Orchestrates the full extraction: builds label map, loops every video, saves one .npy per video
def main():
    LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "holistic_landmarker.task"
    ensure_model(HOLISTIC_MODEL_URL, model_path)

    _, idx_to_word = load_class_list(RAW_DIR)
    video_label_map = load_video_label_map(RAW_DIR)

    used_indices = set(video_label_map.values())
    label_map = {idx_to_word[i]: i for i in sorted(used_indices) if i in idx_to_word}
    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Saved {len(label_map)} class labels -> {LABELS_PATH}")

    videos_dir = RAW_DIR / "videos"
    saved, skipped, missing = 0, 0, 0

    opts = mp_vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
    )

    with mp_vision.HolisticLandmarker.create_from_options(opts) as detector:
        for vid_id, label_idx in tqdm(video_label_map.items(), desc="Videos"):
            video_path = videos_dir / f"{vid_id}.mp4"
            if not video_path.exists():
                missing += 1
                continue

            word     = idx_to_word.get(label_idx, f"class_{label_idx}")
            out_path = LANDMARKS_DIR / f"{word}_{vid_id}.npy"

            if out_path.exists():
                skipped += 1
                continue

            np.save(out_path, process_video(video_path, detector))
            saved += 1

    print(f"Done. Saved: {saved}  Skipped: {skipped}  Missing videos: {missing}")


if __name__ == "__main__":
    main()
