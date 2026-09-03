import torch
from PIL import Image
from torch import nn

from src.predict import AgePredictor


class ConstantModel(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[self.value]])


def regression_predictor(value: float) -> AgePredictor:
    predictor = AgePredictor.__new__(AgePredictor)
    predictor.task = "regression"
    predictor.model = ConstantModel(value)
    predictor.transform = lambda image: torch.zeros(3, 8, 8)
    return predictor


def test_regression_prediction_reports_benchmark_error() -> None:
    result = regression_predictor(31.46).predict(Image.new("RGB", (8, 8)))

    assert result == {
        "predicted_age": 31.5,
        "typical_error_years": 5.2,
        "p90_error_years": 11.5,
    }


def test_regression_prediction_is_clamped_to_supported_ages() -> None:
    assert regression_predictor(-10).predict(Image.new("RGB", (8, 8)))["predicted_age"] == 0
    assert regression_predictor(150).predict(Image.new("RGB", (8, 8)))["predicted_age"] == 116
