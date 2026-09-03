import numpy as np
import pytest
from PIL import Image

from src.face_detection import FaceDetectionError, crop_detected_face


class StubDetector:
    def __init__(self, faces: list[tuple[int, int, int, int]]) -> None:
        self.faces = np.asarray(faces, dtype=np.float32).reshape((-1, 4))
        self.input_size = None

    def setInputSize(self, size) -> None:
        self.input_size = size

    def detect(self, image):
        faces = self.faces if len(self.faces) else None
        return 1, faces


def test_detected_face_is_cropped_with_padding() -> None:
    image = Image.new("RGB", (100, 80), "white")
    detector = StubDetector([(20, 10, 40, 40)])

    face = crop_detected_face(image, detector=detector)

    assert detector.input_size == (100, 80)
    assert face.mode == "RGB"
    assert face.size == (56, 56)


@pytest.mark.parametrize(
    ("faces", "message"),
    [
        ([], "No face detected"),
        ([(5, 5, 20, 20), (40, 5, 20, 20)], "Multiple faces detected"),
    ],
)
def test_upload_requires_exactly_one_face(faces, message) -> None:
    detector = StubDetector(faces)

    with pytest.raises(FaceDetectionError, match=message):
        crop_detected_face(Image.new("RGB", (80, 80)), detector=detector)
