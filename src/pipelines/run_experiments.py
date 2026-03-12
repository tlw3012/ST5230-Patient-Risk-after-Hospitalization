"""
Pipelines for running modeling experiments.

Planned responsibilities:
- load prepared feature matrices and labels from `src.features`
- configure and invoke training routines in `src.models`
- evaluate trained models and collect metrics using `src.models.evaluation`
- optionally coordinate multiple experiments (e.g., different labels
  such as mortality, severity, or acuity) in a repeatable way

This module is intended to be called from the command line or from
lightweight notebooks that focus on visualization and narrative.
"""

