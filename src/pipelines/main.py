"""CLI entry: load features CSV, train logistic model, print metrics (no notebooks)."""

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
    """Run one logistic experiment from a features CSV; returns metrics."""
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
    parser = argparse.ArgumentParser(description="Logistic experiment from features CSV.")
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
    print("\nSummary metrics:", metrics.as_dict())


if __name__ == "__main__":
    main()

