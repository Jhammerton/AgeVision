"""Face detection and cropping for user-uploaded portraits."""

from functools import lru_cache
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from PIL import Image

YUNET_MODEL = Path(__file__).parent / "assets" / "face_detection_yunet_2023mar.onnx"
_DETECTION_LOCK = Lock()


class FaceDetectionError(ValueError):
    """Raised when an upload does not contain exactly one detectable face."""


@lru_cache(maxsize=1)
def load_face_detector():
    if not YUNET_MODEL.exists():
        raise RuntimeError("The YuNet face detector model is not installed")
    return cv2.FaceDetectorYN.create(
        str(YUNET_MODEL),
        "",
        (320, 320),
        score_threshold=0.7,
        nms_threshold=0.3,
        top_k=5000,
    )


def crop_detected_face(
    image: Image.Image,
    detector=None,
    padding_ratio: float = 0.2,
) -> Image.Image:
    """Detect exactly one face and return a padded RGB crop."""
    rgb_image = image.convert("RGB")
    pixels = np.asarray(rgb_image)
    active_detector = detector or load_face_detector()
    with _DETECTION_LOCK:
        active_detector.setInputSize(rgb_image.size)
        _, faces = active_detector.detect(pixels)

    if faces is None or len(faces) == 0:
        raise FaceDetectionError("No face detected; upload a clear, front-facing portrait")
    if len(faces) > 1:
        raise FaceDetectionError("Multiple faces detected; upload an image with one person")

    x, y, width, height = (int(value) for value in faces[0][:4])
    padding = round(max(width, height) * padding_ratio)
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(rgb_image.width, x + width + padding)
    bottom = min(rgb_image.height, y + height + padding)
    return rgb_image.crop((left, top, right, bottom))
