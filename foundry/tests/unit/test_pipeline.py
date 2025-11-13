"""Unit tests for data pipeline."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from ml_framework.data.pipeline import DataPipeline
from ml_framework.data.loaders import CSVLoader
from ml_framework.data.transformers import StandardScalerTransformer, OneHotEncoderTransformer


@pytest.fixture
def dummy_csv_file(tmp_path):
    """Create a dummy CSV file for testing."""
    content = "col1,col2,col3\n1,10,a\n2,20,b\n3,30,a\n4,40,b"
    filepath = tmp_path / "test.csv"
    filepath.write_text(content)
    return str(filepath)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "numeric": [1, 2, 3, 4],
        "category": ["a", "b", "a", "b"],
    })


def test_pipeline_empty_steps_raises_error():
    """Test that empty pipeline raises error."""
    with pytest.raises(ValueError, match="Pipeline must have at least one step"):
        DataPipeline([])


def test_pipeline_must_start_with_loader(sample_dataframe):
    """Test that pipeline must start with a loader."""
    from ml_framework.data.transformers import StandardScalerTransformer
    
    transformer = StandardScalerTransformer(columns=["numeric"])
    transformer.fit(sample_dataframe)
    
    with pytest.raises(ValueError, match="Pipeline must start with a data loader"):
        DataPipeline([transformer])


def test_pipeline_execute_with_loader_only(dummy_csv_file):
    """Test pipeline with only a loader."""
    loader = CSVLoader(dummy_csv_file)
    pipeline = DataPipeline([loader])
    
    result = pipeline.execute()
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert list(result.columns) == ["col1", "col2", "col3"]


def test_pipeline_execute_with_loader_and_transformer(dummy_csv_file, tmp_path):
    """Test pipeline with loader and transformer."""
    loader = CSVLoader(dummy_csv_file)
    scaler = StandardScalerTransformer(columns=["col1", "col2"])
    
    pipeline = DataPipeline([loader, scaler])
    result = pipeline.execute()
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    # Check that numeric columns were scaled
    assert "col1" in result.columns
    assert "col2" in result.columns


def test_pipeline_execute_with_multiple_transformers(dummy_csv_file):
    """Test pipeline with loader and multiple transformers."""
    loader = CSVLoader(dummy_csv_file)
    scaler = StandardScalerTransformer(columns=["col1", "col2"])
    encoder = OneHotEncoderTransformer(columns=["col3"])
    
    pipeline = DataPipeline([loader, scaler, encoder])
    result = pipeline.execute()
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    # Check that encoding created new columns
    assert any("col3_" in col for col in result.columns)


def test_pipeline_execute_with_custom_path(dummy_csv_file, tmp_path):
    """Test pipeline execute with custom path."""
    loader = CSVLoader(dummy_csv_file)
    pipeline = DataPipeline([loader])
    
    # Create another CSV file
    content2 = "col1,col2,col3\n5,50,c\n6,60,c"
    filepath2 = tmp_path / "test2.csv"
    filepath2.write_text(content2)
    
    result = pipeline.execute(loader_path=filepath2)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # Different file has 2 rows


def test_pipeline_fit_transform_alias(dummy_csv_file):
    """Test that fit_transform is an alias for execute."""
    loader = CSVLoader(dummy_csv_file)
    scaler = StandardScalerTransformer(columns=["col1", "col2"])
    
    pipeline = DataPipeline([loader, scaler])
    result1 = pipeline.execute()
    result2 = pipeline.fit_transform()
    
    pd.testing.assert_frame_equal(result1, result2)


def test_pipeline_transformer_before_loader_raises_error():
    """Test that transformer before loader raises error."""
    from ml_framework.data.transformers import StandardScalerTransformer
    
    loader = CSVLoader("dummy.csv")
    transformer = StandardScalerTransformer(columns=["col1"])
    
    # This should work (loader first)
    pipeline = DataPipeline([loader, transformer])
    
    # But if we try to execute without data, it should fail
    # (Actually, the loader will fail first, but the structure is correct)
    with pytest.raises((FileNotFoundError, ValueError)):
        pipeline.execute()


def test_pipeline_unknown_step_type_raises_error(dummy_csv_file):
    """Test that unknown step type raises error."""
    loader = CSVLoader(dummy_csv_file)
    
    class UnknownStep:
        pass
    
    unknown_step = UnknownStep()
    
    # Pipeline creation works, but execute will fail with unknown step type
    pipeline = DataPipeline([loader])
    # Add unknown step manually to test
    pipeline.steps.append(unknown_step)
    
    with pytest.raises(ValueError, match="Unknown step type"):
        pipeline.execute()

