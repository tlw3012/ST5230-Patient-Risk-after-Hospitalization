"""
Model evaluation utilities used across CP* and P* notebooks.

The goal of this module is to centralise low-risk, repeated logic for:

- computing standard classification metrics (accuracy, AUROC, reports)
- optionally computing mean squared error (for binary setups)
- printing a consistent summary during experimentation

This keeps notebooks readable while avoiding copy/paste of the same
evaluation code blocks. Behaviour is intentionally very close to the
original notebook snippets.
"""

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
    """
    Container for common binary classification metrics.

    This mirrors the information that was previously printed directly in
    the notebooks (accuracy, AU-ROC, classification report, optional
    mean squared error).
    """

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
    """
    Compute standard binary-classification metrics.

    Parameters
    ----------
    y_true, y_pred
        True and predicted labels.
    y_proba
        Optional array of predicted probabilities for the positive
        class. If provided, AU-ROC is computed as in the notebooks.
    compute_mse
        If True, also compute mean squared error between ``y_true`` and
        ``y_pred`` (or ``y_proba`` when available), matching the
        existing notebook behaviour.
    """

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    auroc: Optional[float]
    if y_proba is not None:
        # Most notebooks use proba for the positive class, e.g.
        # roc_auc_score(y_test, y_pred_prob).
        auroc = roc_auc_score(y_true, y_proba)
    else:
        auroc = None

    mse: Optional[float] = None
    if compute_mse:
        # In existing notebooks, some experiments pass predicted labels,
        # others pass probabilities; both are supported here.
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
    """
    Print a human-readable summary and return the metrics object.

    This function is intended for use inside notebooks as a direct
    replacement for repeated print blocks such as:

    - accuracy
    - classification report
    - AU-ROC score
    - mean squared error
    """

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

