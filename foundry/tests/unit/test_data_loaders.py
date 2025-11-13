"""Unit tests for data loaders."""

import pytest
import pandas as pd
from pathlib import Path

from ml_framework.data.loaders.base_loader import BaseLoader
from ml_framework.data.loaders.csv_loader import CSVLoader


class TestBaseLoader:
    """Tests for BaseLoader abstract class."""
    
    def test_base_loader_is_abstract(self) -> None:
        """Test that BaseLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLoader()  # type: ignore
    
    def test_base_loader_has_load_method(self) -> None:
        """Test that BaseLoader defines load method signature."""
        # Check that load method exists and is abstract
        assert hasattr(BaseLoader, "load")
        assert BaseLoader.load.__isabstractmethod__  # type: ignore


class TestCSVLoader:
    """Tests for CSVLoader implementation."""
    
    def test_csv_loader_initialization(self, sample_csv_file: Path) -> None:
        """Test CSVLoader can be initialized with path."""
        loader = CSVLoader(sample_csv_file)
        assert loader.path == Path(sample_csv_file)
    
    def test_csv_loader_loads_data(self, sample_csv_file: Path) -> None:
        """Test CSVLoader loads CSV file correctly."""
        loader = CSVLoader(sample_csv_file)
        df = loader.load()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "ID" in df.columns
        assert "ALEXA_RANK" in df.columns
        assert "INDUSTRY" in df.columns
    
    def test_csv_loader_with_custom_path(self, sample_csv_file: Path) -> None:
        """Test CSVLoader can load different file via load() method."""
        loader = CSVLoader("dummy_path.csv")
        df = loader.load(sample_csv_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
    
    def test_csv_loader_with_kwargs(self, tmp_path: Path) -> None:
        """Test CSVLoader accepts pandas read_csv kwargs."""
        # Create CSV with custom separator
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("ID;Value\n1;10\n2;20")
        
        loader = CSVLoader(csv_path, sep=";")
        df = loader.load()
        
        assert len(df) == 2
        assert "ID" in df.columns
        assert "Value" in df.columns
    
    def test_csv_loader_file_not_found(self) -> None:
        """Test CSVLoader raises FileNotFoundError for missing file."""
        loader = CSVLoader("nonexistent_file.csv")
        
        with pytest.raises(FileNotFoundError):
            loader.load()
    
    def test_csv_loader_path_is_directory(self, tmp_path: Path) -> None:
        """Test CSVLoader raises ValueError if path is directory."""
        loader = CSVLoader(tmp_path)
        
        with pytest.raises(ValueError, match="Path is not a file"):
            loader.load()
    
    def test_csv_loader_empty_file(self, tmp_path: Path) -> None:
        """Test CSVLoader handles empty CSV file."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")
        
        loader = CSVLoader(empty_csv)
        
        with pytest.raises(pd.errors.EmptyDataError):
            loader.load()
    
    def test_csv_loader_inherits_from_base_loader(self) -> None:
        """Test CSVLoader properly inherits from BaseLoader."""
        assert issubclass(CSVLoader, BaseLoader)
        loader = CSVLoader("test.csv")
        assert isinstance(loader, BaseLoader)

