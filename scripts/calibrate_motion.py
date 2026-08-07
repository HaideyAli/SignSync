"""Measures hand-motion levels on YOUR camera and writes segmenter thresholds.

The segmenter's defaults were tuned against stored recordings, where "rest" is
a frozen frame with exactly zero motion. A real camera never looks like that,
so the thresholds have to come from real measurements — this takes about 20
seconds and writes data/motion_calib.json, which segmenter.py picks up.

    python scripts/calibrate_motion.py

Phase 1: sit as you would between signs (hands wherever they naturally rest).
Phase 2: sign words continuously, as in the demo.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from camera import open_camera
from extract_landmarks import landmarks_from_frame
from holistic import holistic_options
from segmenter import hand_motion

OUT_PATH   = Path("data/motion_calib.json")
REST_SECS  = 8
SIGN_SECS  = 14
W, H = 640, 480


def collect(detector, cap, seconds, label, colour):
    motions, hands_seen, frames = [], 0, 0
    prev = None
    end = time.time() + seconds
    while time.time() < end:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (W, H))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        lm = landmarks_from_frame(
            detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)))
        if prev is not None:
            motions.append(hand_motion(prev, lm))
        hands_seen += int(np.abs(lm[:126]).sum() > 1e-6)
        frames += 1
        prev = lm

        disp = cv2.flip(small, 1)
        cv2.putText(disp, f"{label}  {end - time.time():.0f}s", (14, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(disp, f"{label}  {end - time.time():.0f}s", (14, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
        cv2.imshow("SignBridge - calibration", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt
    return np.array(motions), (hands_seen / max(frames, 1))


def main():
    cap = open_camera(0)
    if not cap.isOpened():
        sys.exit("cannot open webcam")
    print("Opening camera and model...")
    try:
        with mp_vision.HolisticLandmarker.create_from_options(holistic_options()) as det:
            for i in (3, 2, 1):
                print(f"  rest phase in {i}...", flush=True); time.sleep(1)
            rest, rest_hands = collect(det, cap, REST_SECS,
                                       "REST - stay as you would between signs", (120, 200, 255))
            for i in (3, 2, 1):
                print(f"  sign phase in {i}...", flush=True); time.sleep(1)
            sign, sign_hands = collect(det, cap, SIGN_SECS,
                                       "SIGN - keep signing words", (120, 255, 140))
    except KeyboardInterrupt:
        print("cancelled")
        return
    finally:
        cap.release(); cv2.destroyAllWindows()

    if len(rest) < 5 or len(sign) < 5:
        sys.exit("not enough frames captured — try again")

    r50, r95 = float(np.percentile(rest, 50)), float(np.percentile(rest, 95))
    s50, s90 = float(np.percentile(sign, 50)), float(np.percentile(sign, 90))
    print(f"\nrest : median {r50:.3f}  p95 {r95:.3f}  hands visible {rest_hands*100:.0f}% of frames")
    print(f"sign : median {s50:.3f}  p90 {s90:.3f}  hands visible {sign_hands*100:.0f}% of frames")
    print(f"separation (sign median / rest p95): {s50 / max(r95, 1e-6):.1f}x")

    # Enter above rest noise but well under typical signing; floor from rest p95
    floor = round(max(r95 * 1.2, 0.05), 4)
    enter = round(max(2.0, min(4.0, (s50 / max(r50, 1e-6)) * 0.35)), 2)
    calib = {"floor": floor, "enter_mult": enter,
             "rest_median": r50, "rest_p95": r95, "sign_median": s50, "sign_p90": s90,
             "rest_hands_frac": rest_hands}
    OUT_PATH.write_text(json.dumps(calib, indent=2))
    print(f"\nwrote {OUT_PATH}: floor={floor} enter_mult={enter}")
    if s50 < r95 * 2:
        print("WARNING: signing motion barely exceeds rest noise — segmentation "
              "will be unreliable; use --mode manual for the demo.")


if __name__ == "__main__":
    main()
