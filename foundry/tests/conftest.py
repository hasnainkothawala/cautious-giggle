"""Pytest configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "target": [0, 1, 0, 1, 0]
    })


@pytest.fixture
def sample_csv_file(tmp_path: Path) -> Path:
    """Create a temporary CSV file for testing."""
    csv_path = tmp_path / "test_data.csv"
    df = pd.DataFrame({
        "ID": [1, 2, 3],
        "ALEXA_RANK": [100, 200, 300],
        "INDUSTRY": ["Tech", "Finance", "Healthcare"]
    })
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def temp_artifacts_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for artifacts."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    yield artifacts_dir
    shutil.rmtree(artifacts_dir, ignore_errors=True)


@pytest.fixture
def sample_training_data(sample_dataframe: pd.DataFrame):
    """Create sample training data (X, y)."""
    X = sample_dataframe.drop(columns=["target"])
    y = sample_dataframe["target"]
    return X, y

