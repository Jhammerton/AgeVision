"""Model evaluation and reporting."""

from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score


def evaluate_model(model: Any, features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    """Return threshold-independent and classification metrics."""
    probabilities = model.predict_proba(features)[:, 1]
    predictions = model.predict(features)
    return {
        "roc_auc": roc_auc_score(target, probabilities),
        "average_precision": average_precision_score(target, probabilities),
        "classification_report": classification_report(target, predictions, output_dict=True),
    }
