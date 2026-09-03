from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from src import api
from src.face_detection import FaceDetectionError


class StubPredictor:
    def __init__(self, checkpoint) -> None:
        self.checkpoint = checkpoint

    def predict(self, image: Image.Image) -> dict[str, object]:
        return {
            "predicted_age": 31.5,
            "typical_error_years": 5.2,
            "p90_error_years": 11.5,
        }


def image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_and_prediction_with_installed_model(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    monkeypatch.setenv("AGEVISION_CHECKPOINT", str(checkpoint))
    monkeypatch.setattr(api, "AgePredictor", StubPredictor)
    monkeypatch.setattr(api, "crop_detected_face", lambda image: image)

    with TestClient(api.app) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "model_loaded": True,
        }
        page = client.get("/")
        response = client.post(
            "/api/v1/predict",
            files={"file": ("portrait.png", image_bytes(), "image/png")},
        )

    assert page.status_code == 200
    assert "Drop a portrait here" in page.text
    assert "Take a picture" in page.text
    assert "getUserMedia" in page.text
    assert response.status_code == 200
    assert response.json() == {
        "predicted_age": 31.5,
        "typical_error_years": 5.2,
        "p90_error_years": 11.5,
    }


def test_invalid_image_is_rejected(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    monkeypatch.setenv("AGEVISION_CHECKPOINT", str(checkpoint))
    monkeypatch.setattr(api, "AgePredictor", StubPredictor)

    with TestClient(api.app) as client:
        response = client.post(
            "/api/v1/predict",
            files={"file": ("portrait.png", b"not an image", "image/png")},
        )

    assert response.status_code == 400


def test_image_without_one_face_is_rejected(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    monkeypatch.setenv("AGEVISION_CHECKPOINT", str(checkpoint))
    monkeypatch.setattr(api, "AgePredictor", StubPredictor)

    def reject_face(image):
        raise FaceDetectionError("No face detected; upload a clear portrait")

    monkeypatch.setattr(api, "crop_detected_face", reject_face)
    with TestClient(api.app) as client:
        response = client.post(
            "/api/v1/predict",
            files={"file": ("portrait.png", image_bytes(), "image/png")},
        )

    assert response.status_code == 400
    assert response.json()["detail"].startswith("No face detected")


def test_prediction_is_unavailable_without_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AGEVISION_CHECKPOINT", str(tmp_path / "missing.pt"))
    monkeypatch.setattr(api, "crop_detected_face", lambda image: image)

    with TestClient(api.app) as client:
        assert client.get("/health").json()["model_loaded"] is False
        response = client.post(
            "/api/v1/predict",
            files={"file": ("portrait.png", image_bytes(), "image/png")},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model checkpoint is not installed"
