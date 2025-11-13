"""Unit tests for artifact management."""

import pytest
import pandas as pd
import json
from pathlib import Path

from ml_framework.artifacts.artifact_store import ArtifactStore
from ml_framework.models import AutoModel


class TestArtifactStore:
    """Tests for ArtifactStore."""
    
    def test_artifact_store_initialization(self, temp_artifacts_dir: Path) -> None:
        """Test ArtifactStore can be initialized."""
        store = ArtifactStore(base_dir=temp_artifacts_dir)
        assert store.base_dir == temp_artifacts_dir
        assert store.base_dir.exists()
    
    def test_artifact_store_save_load_model(
        self, temp_artifacts_dir: Path, sample_training_data
    ) -> None:
        """Test ArtifactStore can save and load models."""
        X_train, y_train = sample_training_data
        model = AutoModel.from_type("logistic_regression")
        model.train(X_train, y_train)
        
        store = ArtifactStore(base_dir=temp_artifacts_dir)
        model_path = store.save_model(model, "test_experiment", "v1.0.0")
        
        assert model_path.exists()
        
        # Load model
        new_model = AutoModel.from_type("logistic_regression")
        store.load_model("test_experiment", "v1.0.0", new_model)
        
        assert new_model.trained
        predictions = new_model.predict(X_train)
        assert len(predictions) == len(y_train)
    
    def test_artifact_store_save_load_metrics(self, temp_artifacts_dir: Path) -> None:
        """Test ArtifactStore can save and load metrics."""
        store = ArtifactStore(base_dir=temp_artifacts_dir)
        metrics = {"accuracy": 0.95, "f1_score": 0.92}
        
        metrics_path = store.save_metrics(metrics, "test_experiment", "v1.0.0")
        
        assert metrics_path.exists()
        
        # Load metrics
        loaded_metrics = store.load_metrics("test_experiment", "v1.0.0")
        
        assert loaded_metrics == metrics
        assert loaded_metrics["accuracy"] == 0.95
    
    def test_artifact_store_save_load_config(self, temp_artifacts_dir: Path) -> None:
        """Test ArtifactStore can save and load configs."""
        store = ArtifactStore(base_dir=temp_artifacts_dir)
        config = {"model_type": "logistic_regression", "num_epochs": 100}
        
        config_path = store.save_config(config, "test_experiment", "v1.0.0")
        
        assert config_path.exists()
        
        # Load config
        loaded_config = store.load_config("test_experiment", "v1.0.0")
        
        assert loaded_config == config
    
    def test_artifact_store_list_experiments(self, temp_artifacts_dir: Path) -> None:
        """Test ArtifactStore can list experiments."""
        store = ArtifactStore(base_dir=temp_artifacts_dir)
        
        # Save some artifacts
        store.save_metrics({"accuracy": 0.9}, "experiment1", "v1.0.0")
        store.save_metrics({"accuracy": 0.95}, "experiment2", "v1.0.0")
        
        experiments = store.list_experiments()
        
        assert "experiment1" in experiments
        assert "experiment2" in experiments
    
    def test_artifact_store_list_versions(self, temp_artifacts_dir: Path) -> None:
        """Test ArtifactStore can list versions."""
        store = ArtifactStore(base_dir=temp_artifacts_dir)
        
        # Save multiple versions
        store.save_metrics({"accuracy": 0.9}, "test_experiment", "v1.0.0")
        store.save_metrics({"accuracy": 0.95}, "test_experiment", "v1.1.0")
        
        versions = store.list_versions("test_experiment")
        
        assert "v1.0.0" in versions
        assert "v1.1.0" in versions
    
    def test_artifact_store_load_nonexistent_model(self, temp_artifacts_dir: Path) -> None:
        """Test ArtifactStore raises error for nonexistent model."""
        store = ArtifactStore(base_dir=temp_artifacts_dir)
        model = AutoModel.from_type("logistic_regression")
        
        with pytest.raises(FileNotFoundError):
            store.load_model("nonexistent", "v1.0.0", model)
    
    def test_artifact_store_load_nonexistent_metrics(self, temp_artifacts_dir: Path) -> None:
        """Test ArtifactStore raises error for nonexistent metrics."""
        store = ArtifactStore(base_dir=temp_artifacts_dir)
        
        with pytest.raises(FileNotFoundError):
            store.load_metrics("nonexistent", "v1.0.0")
    
    def test_artifact_store_directory_structure(self, temp_artifacts_dir: Path) -> None:
        """Test ArtifactStore creates proper directory structure."""
        store = ArtifactStore(base_dir=temp_artifacts_dir)
        store.save_metrics({"accuracy": 0.9}, "test_experiment", "v1.0.0")
        
        # Check directory structure
        experiment_dir = temp_artifacts_dir / "test_experiment" / "v1.0.0"
        assert experiment_dir.exists()
        assert (experiment_dir / "metrics.json").exists()

