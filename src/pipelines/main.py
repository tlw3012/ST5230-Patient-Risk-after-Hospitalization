"""Entry point for running a simple end-to-end modeling flow without notebooks.

This script shows how the project modules fit together:

- load a pre-built feature table from CSV
- split into train / test
- train a logistic regression model
- evaluate with common metrics

It is intentionally generic: it does not hard-code label definitions or
feature selection rules (those still live in notebooks or calling code).
"""

from __future__ import annotations

import argparse
from typing import Sequence

import pandas as pd

from src.models.evaluation import BinaryClassificationMetrics
from src.models.logistic import create_logistic_regression
from src.models.training import standard_train_test_split
from src.pipelines.run_experiments import run_binary_experiment


def run_logistic_pipeline(
    features_path: str,
    label_column: str,
    *,
    drop_columns: Sequence[str] = (),
) -> BinaryClassificationMetrics:
    """
    Run a single logistic-regression experiment from a features CSV.

    Parameters
    ----------
    features_path:
        Path to a CSV file containing both features and the label
        column. This is typically produced by a notebook or a dedicated
        feature-building step.
    label_column:
        Name of the target column in the CSV.
    drop_columns:
        Optional columns to drop before modeling (e.g. IDs).
    """

    df = pd.read_csv(features_path)

    if label_column not in df.columns:
        raise ValueError(f"label column '{label_column}' not found in {features_path}")

    X = df.drop(columns=[label_column, *drop_columns])
    y = df[label_column]

    model = create_logistic_regression(
        max_iter=100,
        solver="liblinear",
        n_jobs=1,
    )

    _, _, _, _, _, metrics = run_binary_experiment(
        model=model,
        X=X,
        y=y,
        test_size=0.2,
        random_state=5230,
        compute_mse=True,
    )

    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simple logistic-regression experiment from a features CSV.",
    )
    parser.add_argument(
        "--features-path",
        required=True,
        help="Path to CSV file with features + label column.",
    )
    parser.add_argument(
        "--label-column",
        required=True,
        help="Name of label column in the CSV.",
    )
    parser.add_argument(
        "--drop-column",
        action="append",
        default=[],
        help="Column to drop before modeling (can be passed multiple times).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = run_logistic_pipeline(
        features_path=args.features_path,
        label_column=args.label_column,
        drop_columns=args.drop_column,
    )
    # `run_binary_experiment` already prints a summary; here we just show
    # a compact dict for quick inspection or logging.
    print("\nSummary metrics:", metrics.as_dict())


if __name__ == "__main__":
    main()

