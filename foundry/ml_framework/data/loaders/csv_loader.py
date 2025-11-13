"""CSV file loader implementation.

Loads data from CSV files into pandas DataFrames.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from ml_framework.data.loaders.base_loader import BaseLoader


class CSVLoader(BaseLoader):
    """Loads CSV files into DataFrames."""
    
    def __init__(self, path: Path | str, **kwargs) -> None:
        """Initialize loader.
        
        Args:
            path: Path to CSV file
            **kwargs: Additional pd.read_csv() arguments
        """
        self.path = Path(path)
        self.read_csv_kwargs = kwargs
    
    def load(self, path: Optional[Path | str] = None) -> pd.DataFrame:
        """Load CSV file.
        
        Args:
            path: Optional path (uses default if not provided)
        
        Returns:
            DataFrame
        """
        file_path = Path(path) if path is not None else self.path
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        try:
            df = pd.read_csv(file_path, **self.read_csv_kwargs)
            return df
        except pd.errors.EmptyDataError as e:
            raise pd.errors.EmptyDataError(f"CSV file is empty: {file_path}") from e
        except pd.errors.ParserError as e:
            raise pd.errors.ParserError(f"Failed to parse CSV file: {file_path}") from e

