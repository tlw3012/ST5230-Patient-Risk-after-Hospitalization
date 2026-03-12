"""
Path configuration utilities.

This module will define helper functions and constants for:
- locating raw data directories (e.g., hospital, ED, ICU, notes)
- locating intermediate data outputs (cleaned CSV and PKL files)
- locating model artifacts and reports

The implementation will intentionally avoid hard-coding machine-specific
absolute paths such as Windows drive letters, and instead rely on
project-relative locations and simple configuration.
"""

