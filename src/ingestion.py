"""Download or load the Diabetes 130-US Hospitals dataset."""

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Load the raw diabetic encounter CSV from disk."""
    source = path or RAW_DATA_DIR / "diabetic_data.csv"
    if not source.exists():
        raise FileNotFoundError(
            f"Dataset not found at {source}. Download the UCI dataset and place "
            "diabetic_data.csv in data/raw/."
        )
    return pd.read_csv(source)


if __name__ == "__main__":
    frame = load_raw_data()
    print(f"Loaded {len(frame):,} encounters with {len(frame.columns)} columns.")
