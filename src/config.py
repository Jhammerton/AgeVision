"""Paths and experiment configuration."""

from dataclasses import dataclass
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "UTKFace"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
AGE_BINS = (0, 13, 20, 30, 40, 50, 60, 120)
AGE_GROUPS = ("0-12", "13-19", "20-29", "30-39", "40-49", "50-59", "60+")


@dataclass(frozen=True)
class Settings:
    task: str = "regression"
    architecture: str = "resnet18"
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    pretrained: bool = True
    random_state: int = 42
    validation_size: float = 0.15
    test_size: float = 0.15
    age_bins: tuple[int, ...] = AGE_BINS


def load_settings(path: Path | None = None) -> Settings:
    values = yaml.safe_load((path or PROJECT_ROOT / "configs" / "model.yaml").read_text())
    values["age_bins"] = tuple(values["age_bins"])
    return Settings(**values)
