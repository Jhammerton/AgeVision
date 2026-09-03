"""PyTorch dataset backed by an AgeVision CSV manifest."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class FaceAgeDataset(Dataset):
    def __init__(self, manifest: str | Path | pd.DataFrame, transform, task="regression"):
        self.frame = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest
        self.transform = transform
        self.task = task

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.path) as source:
            image = self.transform(source.convert("RGB"))
        target = float(row.age) if self.task == "regression" else int(row.age_group)
        dtype = torch.float32 if self.task == "regression" else torch.long
        return image, torch.tensor(target, dtype=dtype)
