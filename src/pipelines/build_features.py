"""
Pipelines for building feature tables and embeddings.

Planned high-level steps (to be implemented later):
- load raw or cleaned clinical tables via `src.data`
- construct wide analysis tables via `src.data.building`
- generate numerical and text-based features via `src.features`
- persist intermediate artifacts for reuse in modeling experiments

This module will not contain heavy business logic itself; instead it
will coordinate calls into the lower-level data and feature modules.
"""

