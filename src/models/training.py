"""
Generic training pipelines for supervised models.

This module will provide:
- a unified `train_model` style interface that accepts a model
  specification, feature matrix, and labels
- optional support for basic hyperparameter sweeps or cross-validation
- simple hooks for logging metrics and saving trained models

Model-specific details (e.g., how to construct a LogisticRegression
instance) will live in dedicated modules such as `logistic.py`.
"""

