"""Train an apparent-age regression or classification model."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)

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

    if architecture == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, outputs)
        return model

    if architecture == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            outputs,
        )
        return model

    raise ValueError(f"Unsupported architecture: {architecture}")

def age_group_sample_weights(age_groups) -> torch.Tensor:
    """Return moderate inverse-square-root weights for age-group sampling."""
    groups = torch.tensor(age_groups, dtype=torch.long)
    counts = torch.bincount(groups)
    return counts[groups].double().rsqrt()


def build_age_group_sampler(
    dataset: FaceAgeDataset,
) -> WeightedRandomSampler:
    weights = age_group_sample_weights(
        dataset.frame["age_group"].to_numpy()
    )
    return WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
    )


def train(
    task: str,
    epochs: int,
    architecture: str,
    output: Path,
    balanced_sampling: bool = False,
) -> None:
    settings = load_settings()
    torch.manual_seed(settings.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(task, architecture, settings.pretrained).to(device)
    train_dataset = FaceAgeDataset(PROCESSED_DATA_DIR / "train.csv",
                                   build_transforms(settings.image_size, True), task)
    validation_dataset = FaceAgeDataset(PROCESSED_DATA_DIR / "validation.csv",
                                        build_transforms(settings.image_size), task)
    loader_options = {
        "batch_size": settings.batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    sampler = (
        build_age_group_sampler(train_dataset)
        if balanced_sampling
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=sampler is None,
        sampler=sampler,
        **loader_options,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    loss_fn = nn.HuberLoss(delta=5.0) if task == "regression" else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), settings.learning_rate,
                                  weight_decay=settings.weight_decay)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    print(
        f"device={device} architecture={architecture} task={task} "
        f"balanced_sampling={balanced_sampling}"
    )

    for epoch in range(epochs):
        model.train()
        training_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predictions = model(images)
            if task == "regression":
                predictions = predictions.squeeze(1)
            loss = loss_fn(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * len(images)

        model.eval()
        validation_error = 0.0
        validation_correct = 0
        with torch.inference_mode():
            for images, targets in validation_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                predictions = model(images)
                if task == "regression":
                    predictions = predictions.squeeze(1).clamp(0, 116)
                    validation_error += torch.abs(predictions - targets).sum().item()
                else:
                    validation_correct += (predictions.argmax(1) == targets).sum().item()

        training_loss /= len(train_dataset)
        validation_metric = (
            validation_error / len(validation_dataset)
            if task == "regression"
            else 1.0 - validation_correct / len(validation_dataset)
        )
        metric_name = "val_mae" if task == "regression" else "val_error"
        print(
            f"epoch={epoch + 1} train_loss={training_loss:.4f} "
            f"{metric_name}={validation_metric:.4f}"
        )

        if validation_metric < best_validation:
            best_validation = validation_metric
            torch.save({"state_dict": model.state_dict(), "task": task,
                        "architecture": architecture, "image_size": settings.image_size,
                        "epoch": epoch + 1, "validation_metric": validation_metric}, output)
            print(f"saved_best={output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("regression", "classification"), default="regression")
    parser.add_argument(
    "--architecture",
    choices=("resnet18", "efficientnet_b0", "small_cnn"),
    default="resnet18",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Moderately oversample underrepresented age groups",
    )
    args = parser.parse_args()
    suffix = "_balanced" if args.balanced_sampling else ""
    output = MODEL_DIR / f"{args.architecture}_{args.task}{suffix}.pt"

    train(
        args.task,
        args.epochs,
        args.architecture,
        output,
        args.balanced_sampling,
    )


if __name__ == "__main__":
    main()
