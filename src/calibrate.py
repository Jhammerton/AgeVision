"""Fit post-training linear age calibration on the validation split."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.config import PROCESSED_DATA_DIR
from src.face_detection import FaceDetectionError, crop_detected_face
from src.predict import AgePredictor


def fit_linear_calibration(actual, predicted) -> dict[str, float]:
    """Map raw predictions to ages using validation-only least squares."""
    slope, intercept = np.polyfit(np.asarray(predicted), np.asarray(actual), 1)
    return {"slope": float(slope), "intercept": float(intercept)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROCESSED_DATA_DIR / "validation.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictor = AgePredictor(args.checkpoint)
    frame = pd.read_csv(args.manifest)
    actual = []
    predicted = []
    failed = 0
    for record in frame.itertuples(index=False):
        with Image.open(record.path) as image:
            try:
                face = crop_detected_face(image)
            except FaceDetectionError:
                failed += 1
                continue
            actual.append(record.age)
            predicted.append(predictor.predict(face)["predicted_age"])

    calibration = fit_linear_calibration(actual, predicted)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    checkpoint["calibration"] = calibration
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(
        f"calibration={calibration} samples={len(actual)} "
        f"face_detection_failures={failed} saved={args.output}"
    )


if __name__ == "__main__":
    main()
