"""MediaPipe HolisticLandmarker setup.

RunningMode.IMAGE is not a default to change casually: extraction used IMAGE,
and VIDEO/LIVE_STREAM apply temporal smoothing that would shift live features
away from the distribution the model was fitted on.
"""
from pathlib import Path

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from extract_landmarks import HOLISTIC_MODEL_URL, ensure_model

MODELS_DIR = Path("data/models")


def holistic_options() -> mp_vision.HolisticLandmarkerOptions:
    """Download the task file if needed and build detector options."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    task = MODELS_DIR / "holistic_landmarker.task"
    ensure_model(HOLISTIC_MODEL_URL, task)
    return mp_vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(task)),
        running_mode=mp_vision.RunningMode.IMAGE)
