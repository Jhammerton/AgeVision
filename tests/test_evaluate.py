import pytest

from src.calibrate import fit_linear_calibration
from src.evaluate import regression_metrics


def test_regression_metrics() -> None:
    metrics = regression_metrics([20, 30, 40], [22, 27, 41])
    assert metrics["mae"] == 2.0
    assert metrics["rmse"] == 2.16
    assert metrics["mean_error"] == 0.0


def test_mean_error_preserves_overestimation_direction() -> None:
    metrics = regression_metrics([20, 25], [28, 32])

    assert metrics["mean_error"] == 7.5


def test_linear_calibration_recovers_known_mapping() -> None:
    calibration = fit_linear_calibration([10, 20, 30], [15, 25, 35])

    assert calibration["slope"] == pytest.approx(1.0)
    assert calibration["intercept"] == pytest.approx(-5.0)
