"""Checkpoint loading and apparent-age inference."""
from pathlib import Path

import torch
from PIL import Image

from src.config import AGE_BINS, AGE_GROUPS, MODEL_DIR
from src.preprocessing import build_transforms
from src.train import build_model

REGRESSION_MAE_YEARS = 5.0
REGRESSION_P90_ERROR_YEARS = 11.6


class AgePredictor:
    def __init__(self, checkpoint_path: Path | None = None):
        checkpoint = torch.load(checkpoint_path or MODEL_DIR / "agevision_regression.pt",
                                map_location="cpu", weights_only=True)
        self.task = checkpoint["task"]
        self.model = build_model(self.task, checkpoint["architecture"], pretrained=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.transform = build_transforms(checkpoint.get("image_size", 224))
        self.calibration = checkpoint.get("calibration")

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict[str, object]:
        output = self.model(self.transform(image.convert("RGB")).unsqueeze(0))
        if self.task == "regression":
            age = max(0.0, min(116.0, float(output.squeeze())))
            if self.calibration:
                age = self.calibration["slope"] * age + self.calibration["intercept"]
                age = max(0.0, min(116.0, age))
            return {
                "predicted_age": round(age, 1),
                "typical_error_years": REGRESSION_MAE_YEARS,
                "p90_error_years": REGRESSION_P90_ERROR_YEARS,
            }
        probabilities = output.softmax(1).squeeze(0)
        index = int(probabilities.argmax())
        return {"age_group": AGE_GROUPS[index], "confidence": round(float(probabilities[index]), 3),
                "estimated_range": [AGE_BINS[index], AGE_BINS[index + 1] - 1]}
