import pandas as pd
import pytest

from src.validation import REQUIRED_COLUMNS, validate_raw_data


def test_validation_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_raw_data(pd.DataFrame({"encounter_id": [1]}))


def test_validation_accepts_minimal_valid_frame() -> None:
    frame = pd.DataFrame({column: [1] for column in REQUIRED_COLUMNS})
    validate_raw_data(frame)
