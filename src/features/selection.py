"""
Selection and sampling utilities for admissions and patients.

This module will hold logic for:
- selecting subsets of hadm_id values based on filtering rules
- reproducing the selection of cohorts used in earlier experiments
  (e.g., the creation of hadm_id_selected.csv)
- optional sampling utilities for train/test splits that operate at
  the admission or patient level rather than row level.
"""

