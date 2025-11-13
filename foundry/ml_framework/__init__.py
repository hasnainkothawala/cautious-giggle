"""ML Framework for Data Scientists.

A production-grade framework that enables data scientists to:
1. Gather and transform data
2. Rapidly build and iterate on ML models
3. Host and deploy models
4. Measure model impact

The framework emphasizes modularity, type safety, extensibility, and production-grade reliability.
"""

__version__ = "0.1.0"

from ml_framework.data.loaders import BaseLoader, CSVLoader
from ml_framework.data.transformers import BaseTransformer
from ml_framework.data.pipeline import DataPipeline
from ml_framework.models import BaseModel, AutoModel, ModelTrainer, TrainingArguments
from ml_framework.artifacts import ArtifactStore
from ml_framework.deployment import BaseDeployer
from ml_framework.impact import MetricsCollector

__all__ = [
    "BaseLoader",
    "CSVLoader",
    "BaseTransformer",
    "DataPipeline",
    "BaseModel",
    "AutoModel",
    "ModelTrainer",
    "TrainingArguments",
    "ArtifactStore",
    "BaseDeployer",
    "MetricsCollector",
]

