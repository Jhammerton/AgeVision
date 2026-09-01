"""Train an apparent-age regression or classification model."""

import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from src.config import MODEL_DIR, PROCESSED_DATA_DIR, load_settings
from src.features import FaceAgeDataset
from src.preprocessing import build_transforms


class SmallAgeCNN(nn.Module):
    def __init__(self, outputs: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(64, outputs),
        )

    def forward(self, inputs):
        return self.network(inputs)


def build_model(task="regression", architecture="resnet18", pretrained=True):
    outputs = 1 if task == "regression" else 7
    if architecture == "small_cnn":
        return SmallAgeCNN(outputs)
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, outputs)
    return model


def train(task: str, epochs: int, architecture: str, output: Path) -> None:
    settings = load_settings()
    torch.manual_seed(settings.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(task, architecture, settings.pretrained).to(device)
    dataset = FaceAgeDataset(PROCESSED_DATA_DIR / "train.csv",
                             build_transforms(settings.image_size, True), task)
    loader = DataLoader(dataset, settings.batch_size, shuffle=True, num_workers=0)
    loss_fn = nn.L1Loss() if task == "regression" else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), settings.learning_rate,
                                  weight_decay=settings.weight_decay)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            predictions = model(images)
            if task == "regression":
                predictions = predictions.squeeze(1)
            loss = loss_fn(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(images)
        print(f"epoch={epoch + 1} loss={total / len(dataset):.4f}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "task": task,
                "architecture": architecture, "image_size": settings.image_size}, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("regression", "classification"), default="regression")
    parser.add_argument("--architecture", choices=("resnet18", "small_cnn"), default="resnet18")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train(args.task, args.epochs, args.architecture, MODEL_DIR / f"agevision_{args.task}.pt")


if __name__ == "__main__":
    main()
