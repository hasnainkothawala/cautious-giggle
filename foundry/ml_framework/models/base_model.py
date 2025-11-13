"""Base model abstract class.

Defines the interface that all models must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


class BaseModel(ABC):
    """Base class for ML models. Subclass this to add new models."""
    
    def __init__(self) -> None:
        """Initialize model."""
        self.trained: bool = False
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train model.
        
        Args:
            X: Training features
            y: Training targets
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict on new data.
        
        Args:
            X: Features
        
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk.
        
        Args:
            path: Where to save
        """
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk.
        
        Args:
            path: Where to load from
        """
        pass

