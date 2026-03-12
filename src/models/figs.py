"""
FIGS model helpers based on the `imodels` package.

This module will provide:
- convenience constructors for FIGSClassifier and FIGSRegressor with
  project-appropriate default hyperparameters
- small helper functions to fit these models and collect predictions

The goal is to isolate the dependency on `imodels` in one place while
keeping the public surface area small and easy to test.
"""

