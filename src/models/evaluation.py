"""Common model-evaluation helpers (accuracy, AUROC, classification reports, optional MSE)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    roc_auc_score,
)


@dataclass
class BinaryClassificationMetrics:
    """Holds accuracy, auroc, classification_report str, optional mse."""

    accuracy: float
    auroc: Optional[float]
    classification_report: str
    mse: Optional[float] = None

    def as_dict(self) -> Dict[str, float]:
        data: Dict[str, float] = {"accuracy": float(self.accuracy)}
        if self.auroc is not None:
            data["auroc"] = float(self.auroc)
        if self.mse is not None:
            data["mse"] = float(self.mse)
        return data


def evaluate_binary_classifier(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
    *,
    compute_mse: bool = False,
) -> BinaryClassificationMetrics:
    """Compute accuracy, (optional) AUROC from y_proba, classification report, optional MSE."""
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    auroc: Optional[float]
    if y_proba is not None:
        auroc = roc_auc_score(y_true, y_proba)
    else:
        auroc = None

    mse: Optional[float] = None
    if compute_mse:
        target = y_proba if y_proba is not None else y_pred
        mse = mean_squared_error(y_true, target)

    return BinaryClassificationMetrics(
        accuracy=acc,
        auroc=auroc,
        classification_report=report,
        mse=mse,
    )


def print_binary_classification_summary(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
    *,
    compute_mse: bool = False,
    prefix: str = "",
) -> BinaryClassificationMetrics:
    """Print accuracy, report, AU-ROC, (optional) MSE; return metrics."""
    metrics = evaluate_binary_classifier(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        compute_mse=compute_mse,
    )

    if prefix:
        print(prefix)

    print("Accuracy:", metrics.accuracy)
    print("\nClassification Report:\n", metrics.classification_report)
    if metrics.auroc is not None:
        print("AU-ROC Score:", metrics.auroc)
    if metrics.mse is not None:
        print("Mean Squared Error:", metrics.mse)

    return metrics

