"""Model development module.

Provides abstractions for model training and evaluation.
"""

from ml_framework.models.base_model import BaseModel
from ml_framework.models.auto import AutoModel
from ml_framework.models.trainer import ModelTrainer, TrainingResult
from ml_framework.models.training_arguments import TrainingArguments
from ml_framework.models.wrappers.sklearn_wrapper import SklearnModelWrapper

__all__ = [
    "BaseModel",
    "AutoModel",
    "ModelTrainer",
    "TrainingResult",
    "TrainingArguments",
    "SklearnModelWrapper",
]

