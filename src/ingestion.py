"""Create reproducible train/validation/test manifests from UTKFace."""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import AGE_BINS, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.validation import parse_utkface_filename


def build_manifest(image_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(image_dir.rglob("*.jpg")):
        try:
            rows.append(parse_utkface_filename(path.resolve()))
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"No valid UTKFace JPG images found in {image_dir}")
    frame = pd.DataFrame(rows)
    frame["age_group"] = pd.cut(frame.age, bins=AGE_BINS, right=False, labels=False)
    return frame


def split_manifest(frame: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    train, remainder = train_test_split(frame, test_size=0.30, random_state=seed,
                                        stratify=frame["age_group"])
    validation, test = train_test_split(remainder, test_size=0.50, random_state=seed,
                                        stratify=remainder["age_group"])
    return {"train": train, "validation": validation, "test": test}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in split_manifest(build_manifest(args.input)).items():
        frame.to_csv(args.output / f"{name}.csv", index=False)
        print(f"{name}: {len(frame):,}")


if __name__ == "__main__":
    main()
