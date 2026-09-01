"""Train and persist the readmission model."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import MODEL_DIR, RANDOM_STATE, TARGET_COLUMN
from src.features import build_preprocessor


def train_model(
    data: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
    output_path: Path | None = None,
) -> Pipeline:
    """Fit a baseline logistic-regression pipeline and save it."""
    features = categorical + numeric
    model = Pipeline([
        ("preprocessor", build_preprocessor(categorical, numeric)),
        ("classifier", LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE)),
    ])
    model.fit(data[features], data[TARGET_COLUMN])
    destination = output_path or MODEL_DIR / "readmission_pipeline.joblib"
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return model
