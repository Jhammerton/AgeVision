"""Batch inference entry points."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import MODEL_DIR


def load_model(path: Path | None = None) -> Any:
    """Load a persisted training pipeline."""
    return joblib.load(path or MODEL_DIR / "readmission_pipeline.joblib")


def predict_readmission(model: Any, encounters: pd.DataFrame) -> pd.DataFrame:
    """Add binary predictions and 30-day readmission probabilities."""
    result = encounters.copy()
    result["readmission_probability"] = model.predict_proba(encounters)[:, 1]
    result["predicted_readmission"] = model.predict(encounters)
    return result
