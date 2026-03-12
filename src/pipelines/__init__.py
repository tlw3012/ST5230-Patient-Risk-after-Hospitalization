"""
High-level experiment pipelines.

This subpackage will orchestrate:
- end-to-end data preparation (from raw tables to feature matrices)
- running one or more modeling experiments for specific labels
- saving artifacts and metrics in a structured way

The intent is that command-line entry points or notebooks will call
into these pipelines rather than reimplementing the glue code.
"""

