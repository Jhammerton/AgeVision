"""Overall and sliced evaluation for age regression predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.config import PROCESSED_DATA_DIR, REPORT_DIR
from src.face_detection import FaceDetectionError, crop_detected_face
from src.predict import AgePredictor


def regression_metrics(actual, predicted) -> dict[str, float]:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    signed_errors = predicted - actual
    errors = np.abs(signed_errors)
    return {
        "mae": round(float(errors.mean()), 3),
        "rmse": round(float(np.sqrt(np.mean(signed_errors**2))), 3),
        "mean_error": round(float(signed_errors.mean()), 3),
        "median_absolute_error": round(float(np.median(errors)), 3),
        "p90_absolute_error": round(float(np.percentile(errors, 90)), 3),
    }

def sliced_metrics(frame: pd.DataFrame, group_columns=("age_group", "gender", "ethnicity")):
    report = {"overall": regression_metrics(frame.age, frame.prediction)}
    for column in group_columns:
        report[f"by_{column}"] = {str(name): {**regression_metrics(group.age, group.prediction),
                                              "samples": len(group)}
                                  for name, group in frame.groupby(column, observed=True)}
    return report

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROCESSED_DATA_DIR / "test.csv")
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "metrics.json")
    parser.add_argument(
        "--detect-faces",
        action="store_true",
        help="Apply the same YuNet face crop used by the web application",
    )
    args = parser.parse_args()
    predictor, frame = AgePredictor(args.checkpoint), pd.read_csv(args.manifest)
    predictions = []
    evaluated_indices = []
    failed_indices = []
    for index, path in frame.path.items():
        with Image.open(path) as image:
            if args.detect_faces:
                try:
                    image = crop_detected_face(image)
                except FaceDetectionError:
                    failed_indices.append(index)
                    continue
            predictions.append(predictor.predict(image)["predicted_age"])
            evaluated_indices.append(index)
    evaluated_frame = frame.loc[evaluated_indices].copy()
    evaluated_frame["prediction"] = predictions
    report = sliced_metrics(evaluated_frame)
    if args.detect_faces:
        failures = frame.loc[failed_indices]
        report["face_detection"] = {
            "total": len(frame),
            "detected": len(evaluated_frame),
            "failed": len(failures),
            "detection_rate": round(len(evaluated_frame) / len(frame), 4),
            "failures_by_age_group": {
                str(name): len(group)
                for name, group in failures.groupby("age_group", observed=True)
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
