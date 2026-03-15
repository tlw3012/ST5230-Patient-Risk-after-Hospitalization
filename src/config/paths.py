"""Path helpers for raw data, cleaned outputs, and model artifacts (project-relative)."""

from __future__ import annotations

from pathlib import Path

def project_root() -> Path:
    """Project root (parent of src). Assumes this file lives in src/config/."""
    return Path(__file__).resolve().parent.parent.parent

def data_dir(name: str = "data") -> Path:
    """Subdir under project root, e.g. data, data/raw, data/cleaned."""
    return project_root() / name
