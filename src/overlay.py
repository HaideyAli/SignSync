"""Landmark skeleton drawn over the outgoing video.

The detector already computes these points every frame to feed the classifier;
this renders what was previously thrown away, which is the whole of the
"computer vision demo" look.

Coordinates need mapping back before they can be drawn. MediaPipe normalises
to its own input image — the 4:3 centre crop (camera.crop_4_3) resized to
640x480 — while the display frame is the full, un-cropped, mirrored frame.
_project() undoes both, and getting it wrong shows up as a skeleton that
drifts sideways from the hands.

mediapipe 0.10.35 removed mp.solutions.drawing_utils, so the connection
topology is spelled out here rather than imported.
"""
import cv2
import numpy as np

ENABLED = True            # cleared by gui_app.py --no-skeleton
POSE = True               # cleared by gui_app.py --no-pose

# Landmark layout inside the 258-value frame (see extract_landmarks.py)
_L_HAND, _R_HAND, _POSE = 0, 63, 126

# MediaPipe hand topology: 0 wrist, then 4 joints per finger, thumb first.
_FINGERS = {
    "thumb":  [(0, 1), (1, 2), (2, 3), (3, 4)],
    "index":  [(0, 5), (5, 6), (6, 7), (7, 8)],
    "middle": [(9, 10), (10, 11), (11, 12)],
    "ring":   [(13, 14), (14, 15), (15, 16)],
    "pinky":  [(0, 17), (17, 18), (18, 19), (19, 20)],
    "palm":   [(5, 9), (9, 13), (13, 17)],
}
# BGR. Warm on the thumb through cool on the pinky, so a closing hand reads
# as a colour sweep rather than a single blob.
_COLOURS = {
    "thumb":  (90, 200, 255),    "index":  (255, 205, 80),
    "middle": (245, 130, 225),   "ring":   (130, 255, 190),
    "pinky":  (120, 165, 255),   "palm":   (235, 235, 235),
}
# Shoulders and arms only. Face and legs are noise for a seated signer, and
# the hip links went too: a signer sitting close has hips below the frame, so
# MediaPipe extrapolates them and the torso renders as two long lines running
# off the bottom edge and straight through the caption.
_POSE_LINKS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
_POSE_JOINTS = (11, 12, 13, 14, 15, 16)
_POSE_COLOUR = (225, 175, 95)      # azure, cool against the warm hand palette
_POSE_MIN_VIS = 0.6
_OUTLINE = (12, 14, 18)


def _project(w: int, h: int, mirrored: bool):
    """Map normalised detector coords onto display pixels."""
    crop_w = min(w, int(round(h * 4 / 3)))
    x0 = (w - crop_w) // 2

    def to_px(nx: float, ny: float) -> tuple[int, int]:
        x = nx * crop_w + x0
        return (int(round(w - 1 - x)) if mirrored else int(round(x)),
                int(round(ny * h)))
    return to_px


def _hand_points(lm: np.ndarray, base: int, to_px):
    """21 pixel points, or None when the hand was not detected.

    A missing hand is all zeros by contract (never drop the frame),
    so the sum test distinguishes it from a real hand at the frame origin.
    """
    block = lm[base:base + 63]
    if not block.any():
        return None
    return [to_px(block[i * 3], block[i * 3 + 1]) for i in range(21)]


def _stroke(frame, a, b, colour, core: int, halo: int) -> None:
    """Dark halo then bright core — reads cleanly over any background without
    the cost of a real blur."""
    cv2.line(frame, a, b, _OUTLINE, halo, cv2.LINE_AA)
    cv2.line(frame, a, b, colour, core, cv2.LINE_AA)


def _draw_hand(frame, pts, core: int, halo: int, dot: int) -> None:
    for name, links in _FINGERS.items():
        colour = _COLOURS[name]
        for i, j in links:
            _stroke(frame, pts[i], pts[j], colour, core, halo)
    # Joints stay smaller than the bones they join. Sized level with the core
    # stroke they read as knuckles; any larger and the white circles swallow
    # the colour, and adjacent fingertips merge into one blob.
    for idx, pt in enumerate(pts):
        r = dot + 1 if idx in (4, 8, 12, 16, 20) else dot
        cv2.circle(frame, pt, r + 1, _OUTLINE, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, r, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_pose(frame, lm: np.ndarray, to_px, core: int, halo: int, dot: int) -> None:
    pose = lm[_POSE:]
    if not pose.any():
        return
    h, w = frame.shape[:2]
    pts, vis = [], []
    for i in range(33):
        o = i * 4
        pts.append(to_px(pose[o], pose[o + 1]))
        vis.append(pose[o + 3])
    for i, j in _POSE_LINKS:
        if vis[i] <= _POSE_MIN_VIS or vis[j] <= _POSE_MIN_VIS:
            continue
        # Clip to the frame rather than skipping the segment. An elbow just
        # below the bottom edge is common when seated; dropping the whole limb
        # made the arm flicker on and off as the joint crossed the boundary,
        # while drawing it unclipped is what sprayed lines across the picture.
        ok, a, b = cv2.clipLine((0, 0, w, h), pts[i], pts[j])
        if ok:
            _stroke(frame, a, b, _POSE_COLOUR, core, halo)
    for i in _POSE_JOINTS:
        x, y = pts[i]
        if vis[i] > _POSE_MIN_VIS and 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (x, y), dot + 1, _OUTLINE, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), dot, (255, 255, 255), -1, cv2.LINE_AA)


def draw_landmarks(frame: np.ndarray, lm: np.ndarray | None,
                   mirrored: bool = True) -> np.ndarray:
    """Draw the skeleton in place. No-op when disabled or nothing detected.

    `lm` is one raw (258,) landmark frame straight off the detector thread —
    not the normalised/velocity features, whose coordinates no longer
    correspond to image space.
    """
    if not ENABLED or lm is None or len(lm) < 258:
        return frame
    h, w = frame.shape[:2]
    to_px = _project(w, h, mirrored)
    # Stroke weights scale with frame height so 720p and 1080p match visually
    s = h / 720.0
    core, halo, dot = max(2, int(4 * s)), max(4, int(7 * s)), max(2, int(2.5 * s))

    # Arms carry more weight than finger bones — they are limbs, not joints,
    # and at hand thickness they read as stray threads across the torso.
    if POSE:
        _draw_pose(frame, lm, to_px, max(3, int(6 * s)), max(5, int(11 * s)),
                   max(3, int(4 * s)))
    for base in (_L_HAND, _R_HAND):
        pts = _hand_points(lm, base, to_px)
        if pts is not None:
            _draw_hand(frame, pts, core, halo, dot)
    return frame
