"""Model wrappers module.

Provides wrappers for existing ML libraries (scikit-learn, XGBoost, etc.).
"""

from ml_framework.models.wrappers.sklearn_wrapper import SklearnModelWrapper

__all__ = [
    "SklearnModelWrapper",
]

