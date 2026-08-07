"""Live webcam sign recognition: rolling buffer + prediction engine.
No Zoom logic here — that belongs in zoom_bridge.py.

    python src/inference.py --checkpoint checkpoints/best_model_transformer_50_v10.pth
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.insert(0, str(Path(__file__).parent))
from extract_landmarks import HOLISTIC_MODEL_URL, ensure_model, landmarks_from_frame
from features import preprocess
from model import build_model

W, H = 640, 480          # must match extraction: fixed size keeps landmark scale consistent
CONF_THRESHOLD = 0.80    # per CLAUDE.md
COOLDOWN_S     = 1.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SignPredictor:
    """Wraps the checkpoint and turns a frame buffer into a (word, confidence)."""

    def __init__(self, checkpoint: str):
        # Checkpoints carry their own config, so nothing has to be re-specified
        ckpt = torch.load(checkpoint, map_location=DEVICE)
        self.arch        = ckpt.get("arch", "transformer")
        self.num_classes = ckpt.get("num_classes", 50)
        label_map        = ckpt["label_map"]
        self.idx_to_word = {v: k for k, v in label_map.items()}

        self.model = build_model(self.arch, num_classes=self.num_classes).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, frames: list[np.ndarray]) -> tuple[str, float]:
        raw = np.array(frames, dtype=np.float32)          # (T, 258)
        x   = preprocess(raw)                             # identical to training
        x   = torch.from_numpy(x).unsqueeze(0).to(DEVICE)  # (1, 30, 456)
        probs = torch.softmax(self.model(x), dim=1)[0]
        idx = int(probs.argmax())
        return self.idx_to_word[idx], float(probs[idx])


def hands_visible(frames: list[np.ndarray], recent: int = 10) -> bool:
    """True if either hand was detected recently — otherwise the engine keeps
    predicting confidently over an empty frame."""
    for f in frames[-recent:]:
        if np.abs(f[:126]).sum() > 1e-6:
            return True
    return False


def draw(canvas, lines, y0=30, dy=32, scale=0.7, color=(255, 255, 255)):
    for i, text in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
        cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--buffer", type=int, default=60,
                   help="frames held for one prediction; should span ~one sign")
    p.add_argument("--every", type=int, default=5, help="predict every N frames")
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()

    predictor = SignPredictor(args.checkpoint)
    print(f"Loaded {predictor.arch} / {predictor.num_classes} classes on {DEVICE}")

    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "holistic_landmarker.task"
    ensure_model(HOLISTIC_MODEL_URL, model_path)

    opts = mp_vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        # IMAGE mode matches extraction; VIDEO/LIVE_STREAM add temporal smoothing
        # and would shift live features away from the training distribution
        running_mode=mp_vision.RunningMode.IMAGE,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("ERROR: cannot open webcam")

    buf: deque = deque(maxlen=args.buffer)
    word, conf, last_emit, emitted = "", 0.0, 0.0, ""
    frame_i, fps, t_fps = 0, 0.0, time.time()

    with mp_vision.HolisticLandmarker.create_from_options(opts) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Detect on the un-flipped frame — mirroring swaps hand identity
            small = cv2.resize(frame, (W, H))
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            buf.append(landmarks_from_frame(
                detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))))

            frame_i += 1
            if frame_i % 10 == 0:
                now   = time.time()
                fps   = 10.0 / max(1e-6, now - t_fps)
                t_fps = now

            if len(buf) == buf.maxlen and frame_i % args.every == 0:
                frames = list(buf)
                if hands_visible(frames):
                    word, conf = predictor.predict(frames)
                else:
                    word, conf = "", 0.0

            now = time.time()
            if conf >= CONF_THRESHOLD and now - last_emit >= COOLDOWN_S and word:
                emitted, last_emit = word, now      # zoom_bridge.send_to_zoom_chat goes here

            canvas = cv2.flip(small, 1)             # mirror for natural viewing only
            fill = len(buf) / buf.maxlen
            draw(canvas, [
                f"{word.upper()}  {conf*100:4.1f}%" if word else "...",
                f"buffer {int(fill*100):3d}%   {fps:4.1f} fps   ({buf.maxlen/max(fps,1):.1f}s span)",
                f"last: {emitted.upper()}" if emitted else "",
            ], color=(0, 255, 120) if conf >= CONF_THRESHOLD else (200, 200, 200))
            draw(canvas, ["Q = quit"], y0=H - 16, scale=0.5, color=(160, 160, 160))

            cv2.imshow("SignBridge — Live", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
