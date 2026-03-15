"""Orchestrate data load + build + features -> feature table (to be implemented)."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from src.data.building import load_cleaned_tables as _load_cleaned


def load_cleaned_tables(
    admissions_path: Optional[str] = None,
    discharge_path: Optional[str] = None,
    drgcodes_path: Optional[str] = None,
    triage_path: Optional[str] = None,
    omr_hadmid_path: Optional[str] = None,
    omr_new_path: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Load the six cleaned tables from paths (defaults: notebook filenames in cwd)."""
    kwargs = {}
    if admissions_path is not None:
        kwargs["admissions_path"] = admissions_path
    if discharge_path is not None:
        kwargs["discharge_path"] = discharge_path
    if drgcodes_path is not None:
        kwargs["drgcodes_path"] = drgcodes_path
    if triage_path is not None:
        kwargs["triage_path"] = triage_path
    if omr_hadmid_path is not None:
        kwargs["omr_hadmid_path"] = omr_hadmid_path
    if omr_new_path is not None:
        kwargs["omr_new_path"] = omr_new_path
    return _load_cleaned(**kwargs)
