"""Schema and data-quality validation."""

import pandas as pd


REQUIRED_COLUMNS = {
    "encounter_id", "patient_nbr", "race", "gender", "age",
    "time_in_hospital", "num_lab_procedures", "num_medications", "readmitted",
}


def validate_raw_data(data: pd.DataFrame) -> None:
    """Raise a clear error when required columns or records are missing."""
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("The dataset contains no rows.")
    if data["encounter_id"].duplicated().any():
        raise ValueError("encounter_id must be unique.")
