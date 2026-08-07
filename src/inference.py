"""Live webcam sign recognition. No Zoom logic here — that is zoom_bridge.py.

    python src/inference.py --checkpoint checkpoints/best_model_transformer_50_v10.pth

SPACE captures one 4-second take and predicts, mirroring how record_signs.py
collected the training data. --auto instead fires on detected hand motion.
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.insert(0, str(Path(__file__).parent))
from extract_landmarks import HOLISTIC_MODEL_URL, ensure_model, landmarks_from_frame
from predictor import (DEVICE, RECORD_SECS, SIGN_ONSET_FRAC, MotionTrigger,
                       SignPredictor, hands_visible)

W, H = 640, 480          # must match extraction so landmark scale is consistent
CONF_THRESHOLD = 0.80    # per CLAUDE.md
COOLDOWN_S     = 1.5
WARMUP_FRAMES  = 30      # measure real fps before sizing the capture window


def txt(img, s, x, y, scale=0.7, color=(255, 255, 255)):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--auto", action="store_true", help="trigger on motion instead of SPACE")
    p.add_argument("--seconds", type=float, default=RECORD_SECS,
                   help="capture length; must match how the data was recorded")
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()

    predictor = SignPredictor(args.checkpoint)
    print(f"Loaded {predictor.arch} / {predictor.num_classes} classes on {DEVICE}")

    models_dir = Path("data/models"); models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "holistic_landmarker.task"
    ensure_model(HOLISTIC_MODEL_URL, model_path)

    opts = mp_vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        # IMAGE mode matches extraction; VIDEO/LIVE_STREAM add temporal smoothing
        running_mode=mp_vision.RunningMode.IMAGE)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("ERROR: cannot open webcam")

    ring: deque = deque(maxlen=600)      # generous raw history
    trigger = MotionTrigger()
    prev_lm = None
    need, results, last_emit = 0, [], 0.0
    frame_i, fps, t0 = 0, 0.0, time.time()
    window = 0                            # frames per capture, set after warmup

    with mp_vision.HolisticLandmarker.create_from_options(opts) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Detect on the un-flipped frame — mirroring swaps hand identity
            small = cv2.resize(frame, (W, H))
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            lm    = landmarks_from_frame(
                detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)))
            ring.append(lm)
            frame_i += 1

            if frame_i <= WARMUP_FRAMES:
                fps = frame_i / max(1e-6, time.time() - t0)
                window = max(10, int(round(args.seconds * fps)))
            elif frame_i % 10 == 0:
                fps = 0.9 * fps + 0.1 * (10 / max(1e-6, time.time() - t0))
                t0 = time.time()
            if frame_i == WARMUP_FRAMES:
                t0 = time.time()
                print(f"~{fps:.1f} fps -> capturing {window} frames per {args.seconds}s take")

            fire = False
            if args.auto and need == 0 and frame_i > WARMUP_FRAMES:
                fire = trigger.update(prev_lm, lm)
            prev_lm = lm

            if fire:
                # Start the window before the sign so it lands ~18% in, as in training
                need = max(1, window - int(window * SIGN_ONSET_FRAC))

            if need > 0:
                need -= 1
                if need == 0 and len(ring) >= window:
                    frames = list(ring)[-window:]
                    if hands_visible(frames, recent=window):
                        results = predictor.predict(frames)
                        if results[0][1] >= CONF_THRESHOLD:
                            last_emit = time.time()   # zoom_bridge.send_to_zoom_chat() here
                    else:
                        results = []

            canvas = cv2.flip(small, 1)   # mirror the display only
            if results:
                word, conf = results[0]
                good = conf >= CONF_THRESHOLD
                txt(canvas, f"{word.upper()}  {conf*100:.0f}%", 12, 44, 1.1,
                    (0, 255, 120) if good else (80, 180, 255))
                alts = "   ".join(f"{w} {c*100:.0f}%" for w, c in results[1:3])
                txt(canvas, alts, 12, 76, 0.55, (170, 170, 170))
            elif need > 0:
                txt(canvas, f"RECORDING  {need/max(fps,1):.1f}s", 12, 44, 1.1, (30, 30, 255))
            else:
                txt(canvas, "SPACE to sign" if not args.auto else "waiting for motion",
                    12, 44, 0.8, (200, 200, 200))

            txt(canvas, f"{fps:.0f} fps   window {window}f/{args.seconds:.1f}s"
                        + ("   AUTO" if args.auto else ""), 12, H - 40, 0.5, (150, 150, 150))
            txt(canvas, "SPACE = capture    Q = quit", 12, H - 16, 0.5, (150, 150, 150))
            cv2.imshow("SignBridge — Live", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and need == 0 and frame_i > WARMUP_FRAMES:
                results = []
                need = window   # capture a full take starting now, like record_signs.py

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
