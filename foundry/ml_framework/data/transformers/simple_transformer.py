"""Simple transformer implementations.

Basic transformers wrapping scikit-learn's preprocessing utilities.
"""

from typing import List, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

from ml_framework.data.transformers.base_transformer import BaseTransformer


class StandardScalerTransformer(BaseTransformer):
    """Wraps sklearn's StandardScaler for DataFrame support."""
    
    def __init__(self, columns: Optional[List[str]] = None) -> None:
        """Initialize scaler.
        
        Args:
            columns: Columns to scale (None = all numeric)
        """
        super().__init__()
        self.columns = columns
        self.scaler: Optional[StandardScaler] = None
        self.numeric_columns: Optional[List[str]] = None
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit scaler on data.
        
        Args:
            data: Training data
        """
        if data.empty:
            raise ValueError("Cannot fit transformer on empty DataFrame")
        
        self.scaler = StandardScaler()
        
        if self.columns:
            # Validate columns exist
            missing_cols = set(self.columns) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Columns not found in data: {missing_cols}")
            self.scaler.fit(data[self.columns])
        else:
            # Auto-detect numeric columns
            self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if not self.numeric_columns:
                raise ValueError("No numeric columns found in data")
            self.scaler.fit(data[self.numeric_columns])
        
        self.fitted = True
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale data.
        
        Args:
            data: Data to transform
        
        Returns:
            Scaled DataFrame
        """
        if not self.fitted or self.scaler is None:
            raise ValueError("Transformer must be fitted before transform()")
        
        if data.empty:
            raise ValueError("Cannot transform empty DataFrame")
        
        data = data.copy()
        
        if self.columns:
            data[self.columns] = self.scaler.transform(data[self.columns])
        else:
            if self.numeric_columns is None:
                raise ValueError("Numeric columns not determined during fit()")
            data[self.numeric_columns] = self.scaler.transform(data[self.numeric_columns])
        
        return data


class MinMaxScalerTransformer(BaseTransformer):
    """Wraps sklearn's MinMaxScaler for DataFrame support."""
    
    def __init__(self, columns: Optional[List[str]] = None, feature_range: tuple = (0, 1)) -> None:
        """Initialize scaler.
        
        Args:
            columns: Columns to scale
            feature_range: Output range
        """
        super().__init__()
        self.columns = columns
        self.feature_range = feature_range
        self.scaler: Optional[MinMaxScaler] = None
        self.numeric_columns: Optional[List[str]] = None
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit scaler."""
        if data.empty:
            raise ValueError("Cannot fit transformer on empty DataFrame")
        
        self.scaler = MinMaxScaler(feature_range=self.feature_range)
        
        if self.columns:
            missing_cols = set(self.columns) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Columns not found in data: {missing_cols}")
            self.scaler.fit(data[self.columns])
        else:
            self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if not self.numeric_columns:
                raise ValueError("No numeric columns found in data")
            self.scaler.fit(data[self.numeric_columns])
        
        self.fitted = True
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale data."""
        if not self.fitted or self.scaler is None:
            raise ValueError("Transformer must be fitted before transform()")
        
        if data.empty:
            raise ValueError("Cannot transform empty DataFrame")
        
        data = data.copy()
        
        if self.columns:
            data[self.columns] = self.scaler.transform(data[self.columns])
        else:
            if self.numeric_columns is None:
                raise ValueError("Numeric columns not determined during fit()")
            data[self.numeric_columns] = self.scaler.transform(data[self.numeric_columns])
        
        return data


class OneHotEncoderTransformer(BaseTransformer):
    """Wraps sklearn's OneHotEncoder for DataFrame support."""
    
    def __init__(self, columns: List[str], drop: Optional[str] = None, sparse_output: bool = False, handle_unknown: str = "ignore") -> None:
        """Initialize encoder.
        
        Args:
            columns: Columns to encode
            drop: Drop strategy
            sparse_output: Return sparse matrix
            handle_unknown: How to handle unknowns
        """
        super().__init__()
        if not columns:
            raise ValueError("columns must be a non-empty list")
        self.columns = columns
        self.drop = drop
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown
        self.encoder: Optional[OneHotEncoder] = None
        self.feature_names: Optional[List[str]] = None
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit encoder."""
        if data.empty:
            raise ValueError("Cannot fit transformer on empty DataFrame")
        
        missing_cols = set(self.columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in data: {missing_cols}")
        
        self.encoder = OneHotEncoder(drop=self.drop, sparse_output=self.sparse_output, handle_unknown=self.handle_unknown)
        self.encoder.fit(data[self.columns])
        
        # Generate feature names
        self.feature_names = self.encoder.get_feature_names_out(self.columns).tolist()
        
        self.fitted = True
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode data."""
        if not self.fitted or self.encoder is None:
            raise ValueError("Transformer must be fitted before transform()")
        
        if data.empty:
            raise ValueError("Cannot transform empty DataFrame")
        
        missing_cols = set(self.columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in data: {missing_cols}")
        
        data = data.copy()
        
        # Encode categorical columns
        encoded = self.encoder.transform(data[self.columns])
        
        # Convert to DataFrame
        if self.feature_names is None:
            raise ValueError("Feature names not determined during fit()")
        encoded_df = pd.DataFrame(encoded, columns=self.feature_names, index=data.index)
        
        # Drop original columns and add encoded columns
        data = data.drop(columns=self.columns)
        data = pd.concat([data, encoded_df], axis=1)
        
        return data

