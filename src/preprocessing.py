"""Cleaning and target construction."""

import pandas as pd

from src.config import TARGET_COLUMN


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize missing values and create a binary 30-day readmission target."""
    cleaned = data.copy()
    cleaned = cleaned.replace("?", pd.NA)
    cleaned[TARGET_COLUMN] = (cleaned["readmitted"] == "<30").astype("int8")
    cleaned = cleaned.drop(columns=["weight", "payer_code", "medical_specialty"], errors="ignore")
    return cleaned
