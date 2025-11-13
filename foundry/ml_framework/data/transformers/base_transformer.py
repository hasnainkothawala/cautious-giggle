"""Base transformer abstract class.

Defines the interface for data transformation using fit/transform pattern.
"""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class BaseTransformer(ABC):
    """Base class for data transformers using fit/transform pattern.
    
    Attributes:
        fitted: Whether transformer has been fitted
    """
    
    def __init__(self) -> None:
        """Initialize transformer."""
        self.fitted: bool = False
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Fit transformer on data.
        
        Args:
            data: Training DataFrame
        """
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation to data.
        
        Args:
            data: DataFrame to transform
        
        Returns:
            Transformed DataFrame
        """
        pass
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.
        
        Args:
            data: DataFrame to process
        
        Returns:
            Transformed DataFrame
        """
        self.fit(data)
        return self.transform(data)

