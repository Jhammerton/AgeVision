from src.evaluate import regression_metrics

def test_regression_metrics() -> None:
    metrics = regression_metrics([20, 30, 40], [22, 27, 41])
    assert metrics["mae"] == 2.0
    assert metrics["rmse"] == 2.16
