"""Base data loader abstract class.

Defines the interface for loading data from various sources into pandas DataFrames.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd


class BaseLoader(ABC):
    """Base class for loading data from various sources. Returns pandas DataFrames."""
    
    @abstractmethod
    def load(self, path: Optional[Path] = None) -> pd.DataFrame:
        """Load data and return as DataFrame.
        
        Args:
            path: Optional path to data source
        
        Returns:
            DataFrame with loaded data
        """
        pass

