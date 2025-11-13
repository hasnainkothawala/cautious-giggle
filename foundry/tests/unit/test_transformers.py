"""Unit tests for data transformers."""

import pytest
import pandas as pd
import numpy as np

from ml_framework.data.transformers.base_transformer import BaseTransformer
from ml_framework.data.transformers.simple_transformer import (
    StandardScalerTransformer,
    MinMaxScalerTransformer,
    OneHotEncoderTransformer,
)


class TestBaseTransformer:
    """Tests for BaseTransformer abstract class."""
    
    def test_base_transformer_is_abstract(self) -> None:
        """Test that BaseTransformer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTransformer()  # type: ignore
    
    def test_base_transformer_has_fit_transform(self) -> None:
        """Test that BaseTransformer has fit_transform convenience method."""
        # Create a simple concrete implementation for testing
        class TestTransformer(BaseTransformer):
            def fit(self, data: pd.DataFrame) -> None:
                self.fitted = True
            
            def transform(self, data: pd.DataFrame) -> pd.DataFrame:
                return data
        
        transformer = TestTransformer()
        df = pd.DataFrame({"col": [1, 2, 3]})
        result = transformer.fit_transform(df)
        
        assert isinstance(result, pd.DataFrame)
        assert transformer.fitted


class TestStandardScalerTransformer:
    """Tests for StandardScalerTransformer."""
    
    def test_standard_scaler_initialization(self) -> None:
        """Test StandardScalerTransformer can be initialized."""
        scaler = StandardScalerTransformer()
        assert scaler.columns is None
        assert not scaler.fitted
    
    def test_standard_scaler_with_columns(self) -> None:
        """Test StandardScalerTransformer with specific columns."""
        scaler = StandardScalerTransformer(columns=["feature1", "feature2"])
        assert scaler.columns == ["feature1", "feature2"]
    
    def test_standard_scaler_fit_transform(self, sample_dataframe: pd.DataFrame) -> None:
        """Test StandardScalerTransformer fit and transform."""
        scaler = StandardScalerTransformer()
        X = sample_dataframe.drop(columns=["target"])
        
        X_scaled = scaler.fit_transform(X)
        
        assert isinstance(X_scaled, pd.DataFrame)
        assert scaler.fitted
        assert X_scaled.shape == X.shape
        # Check that values are standardized (mean ~0, std ~1)
        assert np.isclose(X_scaled["feature1"].mean(), 0, atol=1e-10)
        assert np.isclose(X_scaled["feature1"].std(), 1, atol=1e-10)
    
    def test_standard_scaler_with_specific_columns(self, sample_dataframe: pd.DataFrame) -> None:
        """Test StandardScalerTransformer with specific columns."""
        scaler = StandardScalerTransformer(columns=["feature1"])
        X = sample_dataframe.drop(columns=["target"])
        
        X_scaled = scaler.fit_transform(X)
        
        # Only feature1 should be scaled
        assert np.isclose(X_scaled["feature1"].mean(), 0, atol=1e-10)
        # feature2 should remain unchanged
        assert X_scaled["feature2"].equals(X["feature2"])
    
    def test_standard_scaler_fit_before_transform(self, sample_dataframe: pd.DataFrame) -> None:
        """Test StandardScalerTransformer requires fit before transform."""
        scaler = StandardScalerTransformer()
        X = sample_dataframe.drop(columns=["target"])
        
        with pytest.raises(ValueError, match="must be fitted"):
            scaler.transform(X)
    
    def test_standard_scaler_empty_dataframe(self) -> None:
        """Test StandardScalerTransformer handles empty DataFrame."""
        scaler = StandardScalerTransformer()
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="empty DataFrame"):
            scaler.fit(empty_df)
    
    def test_standard_scaler_missing_columns(self, sample_dataframe: pd.DataFrame) -> None:
        """Test StandardScalerTransformer raises error for missing columns."""
        scaler = StandardScalerTransformer(columns=["nonexistent"])
        X = sample_dataframe.drop(columns=["target"])
        
        with pytest.raises(ValueError, match="not found in data"):
            scaler.fit(X)
    
    def test_standard_scaler_no_numeric_columns(self) -> None:
        """Test StandardScalerTransformer handles data with no numeric columns."""
        scaler = StandardScalerTransformer()
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        
        with pytest.raises(ValueError, match="No numeric columns"):
            scaler.fit(df)


class TestMinMaxScalerTransformer:
    """Tests for MinMaxScalerTransformer."""
    
    def test_minmax_scaler_fit_transform(self, sample_dataframe: pd.DataFrame) -> None:
        """Test MinMaxScalerTransformer fit and transform."""
        scaler = MinMaxScalerTransformer()
        X = sample_dataframe.drop(columns=["target"])
        
        X_scaled = scaler.fit_transform(X)
        
        assert isinstance(X_scaled, pd.DataFrame)
        assert scaler.fitted
        # Check that values are in [0, 1] range
        assert (X_scaled["feature1"] >= 0).all()
        assert (X_scaled["feature1"] <= 1).all()
    
    def test_minmax_scaler_custom_range(self, sample_dataframe: pd.DataFrame) -> None:
        """Test MinMaxScalerTransformer with custom feature range."""
        scaler = MinMaxScalerTransformer(feature_range=(-1, 1))
        X = sample_dataframe.drop(columns=["target"])
        
        X_scaled = scaler.fit_transform(X)
        
        # Check that values are in [-1, 1] range
        assert (X_scaled["feature1"] >= -1).all()
        assert (X_scaled["feature1"] <= 1).all()


class TestOneHotEncoderTransformer:
    """Tests for OneHotEncoderTransformer."""
    
    def test_onehot_encoder_initialization(self) -> None:
        """Test OneHotEncoderTransformer can be initialized."""
        encoder = OneHotEncoderTransformer(columns=["category"])
        assert encoder.columns == ["category"]
        assert not encoder.fitted
    
    def test_onehot_encoder_empty_columns(self) -> None:
        """Test OneHotEncoderTransformer raises error for empty columns."""
        with pytest.raises(ValueError, match="non-empty list"):
            OneHotEncoderTransformer(columns=[])
    
    def test_onehot_encoder_fit_transform(self) -> None:
        """Test OneHotEncoderTransformer fit and transform."""
        encoder = OneHotEncoderTransformer(columns=["INDUSTRY"])
        df = pd.DataFrame({
            "INDUSTRY": ["Tech", "Finance", "Tech", "Healthcare"],
            "feature1": [1, 2, 3, 4]
        })
        
        df_encoded = encoder.fit_transform(df)
        
        assert isinstance(df_encoded, pd.DataFrame)
        assert encoder.fitted
        # Original column should be removed
        assert "INDUSTRY" not in df_encoded.columns
        # Encoded columns should be added
        assert any("INDUSTRY" in col for col in df_encoded.columns)
        # feature1 should remain
        assert "feature1" in df_encoded.columns
    
    def test_onehot_encoder_fit_before_transform(self) -> None:
        """Test OneHotEncoderTransformer requires fit before transform."""
        encoder = OneHotEncoderTransformer(columns=["INDUSTRY"])
        df = pd.DataFrame({"INDUSTRY": ["Tech", "Finance"]})
        
        with pytest.raises(ValueError, match="must be fitted"):
            encoder.transform(df)
    
    def test_onehot_encoder_missing_columns(self) -> None:
        """Test OneHotEncoderTransformer raises error for missing columns."""
        encoder = OneHotEncoderTransformer(columns=["nonexistent"])
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        
        with pytest.raises(ValueError, match="not found in data"):
            encoder.fit(df)

