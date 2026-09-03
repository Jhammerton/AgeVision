import pytest
from torch import nn
from torchvision.models import EfficientNet, ResNet

from src.train import SmallAgeCNN, build_model


@pytest.mark.parametrize(
    ("architecture", "expected_type"),
    [
        ("small_cnn", SmallAgeCNN),
        ("resnet18", ResNet),
        ("efficientnet_b0", EfficientNet),
    ],
)
def test_regression_model_has_one_output(
    architecture: str,
    expected_type: type[nn.Module],
) -> None:
    model = build_model("regression", architecture, pretrained=False)

    assert isinstance(model, expected_type)

    final_layers = [
        layer for layer in model.modules()
        if isinstance(layer, nn.Linear)
    ]
    assert final_layers[-1].out_features == 1


def test_unsupported_architecture_raises_error() -> None:
    with pytest.raises(ValueError, match="Unsupported architecture"):
        build_model("regression", "unknown", pretrained=False)