"""Data loaders module.

Provides abstractions for loading data from various sources.
"""

from ml_framework.data.loaders.base_loader import BaseLoader
from ml_framework.data.loaders.csv_loader import CSVLoader

__all__ = [
    "BaseLoader",
    "CSVLoader",
]

