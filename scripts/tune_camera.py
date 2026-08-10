"""Live camera tuning: adjust the picture while watching it, then copy the flags.

Guessing at exposure/gain/brightness/gamma one round-trip at a time is slow and
subjective. Drag the sliders until it looks right and press S — it prints the
exact gui_app.py command for the settings on screen.

    python scripts/tune_camera.py

Keys:  A auto/manual exposure   S print flags   R reset   Q quit
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from camera import (DEFAULT_BRIGHTNESS, DEFAULT_EXPOSURE, DEFAULT_GAIN,
                    DEFAULT_GAMMA, gamma_lut, open_camera)

WIN = "SignBridge - camera tuning"
DEFAULTS = {"gain": int(DEFAULT_GAIN), "brightness": int(DEFAULT_BRIGHTNESS),
            "gamma x100": int(DEFAULT_GAMMA * 100), "-exposure": int(-DEFAULT_EXPOSURE)}


def make_sliders():
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1100, 700)
    cv2.createTrackbar("gain", WIN, DEFAULTS["gain"], 255, lambda v: None)
    cv2.createTrackbar("brightness", WIN, DEFAULTS["brightness"], 255, lambda v: None)
    cv2.createTrackbar("gamma x100", WIN, DEFAULTS["gamma x100"], 150, lambda v: None)
    # exposure is negative; the slider holds its magnitude (higher = darker/faster)
    cv2.createTrackbar("-exposure", WIN, DEFAULTS["-exposure"], 9, lambda v: None)


def read_sliders():
    g = cv2.getTrackbarPos("gain", WIN)
    b = cv2.getTrackbarPos("brightness", WIN)
    gamma = max(cv2.getTrackbarPos("gamma x100", WIN), 10) / 100.0
    exp = -max(cv2.getTrackbarPos("-exposure", WIN), 1)
    return g, b, gamma, exp


def txt(img, s, y, colour=(255, 255, 255), scale=0.6):
    cv2.putText(img, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1)


def main():
    width, height = 1280, 720          # tune at 720p; the settings carry to 1080p
    auto = False
    cap = open_camera(0, width, height, 30.0, DEFAULT_EXPOSURE, DEFAULT_GAIN,
                      DEFAULT_BRIGHTNESS)
    if not cap.isOpened():
        sys.exit("cannot open webcam")
    make_sliders()

    prev = (None, None, None)
    fps, n, t0, last = 0.0, 0, time.time(), None
    while True:
        gain, bright, gamma, exposure = read_sliders()
        if (gain, bright, exposure) != prev:       # only touch the driver on change
            if not auto:
                cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                cap.set(cv2.CAP_PROP_GAIN, gain)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, bright)
            prev = (gain, bright, exposure)

        ok, frame = cap.read()
        if not ok:
            break
        lut = gamma_lut(gamma)
        shown = cv2.LUT(frame, lut) if lut is not None else frame

        n += 1
        if n % 10 == 0:
            now = time.time(); fps = 10 / max(1e-6, now - t0); t0 = now

        f32 = shown.astype(np.float32)
        grain = 0.0 if last is None else float(np.abs(f32 - last).mean())
        last = f32
        view = cv2.flip(shown, 1).copy()
        mode = "AUTO exposure (~15fps)" if auto else f"manual exposure {exposure}"
        txt(view, f"{mode}   {fps:4.1f} fps", 30, (0, 255, 140), 0.7)
        txt(view, f"gain {gain}  brightness {bright}  gamma {gamma:.2f}", 60)
        txt(view, f"mean {f32.mean():5.1f}   contrast {f32.std():5.1f}   "
                  f"blacks {np.percentile(f32, 5):5.1f}   grain {grain:4.1f}", 88,
            (170, 200, 255), 0.55)
        txt(view, "A auto/manual   S print flags   R reset   Q quit",
            view.shape[0] - 16, (160, 160, 160), 0.5)
        cv2.imshow(WIN, view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            auto = not auto
            cap.release()
            cap = open_camera(0, width, height, 30.0,
                              None if auto else exposure,
                              None if auto else gain,
                              None if auto else bright)
            prev = (None, None, None)
        if key == ord("r"):
            for name, val in DEFAULTS.items():
                cv2.setTrackbarPos(name, WIN, val)
        if key == ord("s"):
            expo = "auto" if auto else str(exposure)
            print("\n  python src/gui_app.py --checkpoint "
                  "checkpoints/best_model_transformer_50_v10.pth \\")
            print(f"      --exposure {expo} --gain {gain} "
                  f"--brightness {bright} --gamma {gamma:.2f}\n", flush=True)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
