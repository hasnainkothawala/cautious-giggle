"""Artifact storage and management.

Handles saving and loading of experiment artifacts (models, metrics, configs).
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import joblib

from ml_framework.models.base_model import BaseModel


class ArtifactStore:
    """Manages storage of experiment artifacts (models, metrics, configs)."""
    
    def __init__(self, base_dir: Path | str = Path("./artifacts")) -> None:
        """Initialize artifact store.
        
        Args:
            base_dir: Base directory for storing artifacts.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_experiment_dir(self, experiment_name: str, version: str) -> Path:
        """Get directory path for experiment version.
        
        Args:
            experiment_name: Name of the experiment.
            version: Version identifier.
        
        Returns:
            Path to experiment version directory.
        """
        experiment_dir = self.base_dir / experiment_name / version
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir
    
    def save_model(self, model: BaseModel, experiment_name: str, version: str) -> Path:
        """Save model weights.
        
        Args:
            model: Model instance to save.
            experiment_name: Name of the experiment.
            version: Version identifier.
        
        Returns:
            Path to saved model file.
        
        Raises:
            IOError: If model cannot be saved.
        """
        experiment_dir = self._get_experiment_dir(experiment_name, version)
        model_path = experiment_dir / "model.pkl"
        
        try:
            model.save(model_path)
            return model_path
        except Exception as e:
            raise IOError(f"Failed to save model to {model_path}") from e
    
    def load_model(self, experiment_name: str, version: str, model: BaseModel) -> None:
        """Load model weights.
        
        Args:
            experiment_name: Name of the experiment.
            version: Version identifier.
            model: Model instance to load weights into.
        
        Raises:
            FileNotFoundError: If model file doesn't exist.
            IOError: If model cannot be loaded.
        """
        experiment_dir = self._get_experiment_dir(experiment_name, version)
        model_path = experiment_dir / "model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            model.load(model_path)
        except Exception as e:
            raise IOError(f"Failed to load model from {model_path}") from e
    
    def save_metrics(self, metrics: Dict[str, Any], experiment_name: str, version: str) -> Path:
        """Save model metrics.
        
        Args:
            metrics: Dictionary of metrics to save.
            experiment_name: Name of the experiment.
            version: Version identifier.
        
        Returns:
            Path to saved metrics file.
        
        Raises:
            IOError: If metrics cannot be saved.
        """
        experiment_dir = self._get_experiment_dir(experiment_name, version)
        metrics_path = experiment_dir / "metrics.json"
        
        try:
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            return metrics_path
        except Exception as e:
            raise IOError(f"Failed to save metrics to {metrics_path}") from e
    
    def load_metrics(self, experiment_name: str, version: str) -> Dict[str, Any]:
        """Load model metrics.
        
        Args:
            experiment_name: Name of the experiment.
            version: Version identifier.
        
        Returns:
            Dictionary of metrics.
        
        Raises:
            FileNotFoundError: If metrics file doesn't exist.
            IOError: If metrics cannot be loaded.
        """
        experiment_dir = self._get_experiment_dir(experiment_name, version)
        metrics_path = experiment_dir / "metrics.json"
        
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
        
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            return metrics
        except Exception as e:
            raise IOError(f"Failed to load metrics from {metrics_path}") from e
    
    def save_config(self, config: Dict[str, Any], experiment_name: str, version: str) -> Path:
        """Save experiment configuration.
        
        Args:
            config: Dictionary of configuration to save.
            experiment_name: Name of the experiment.
            version: Version identifier.
        
        Returns:
            Path to saved config file.
        
        Raises:
            IOError: If config cannot be saved.
        """
        experiment_dir = self._get_experiment_dir(experiment_name, version)
        config_path = experiment_dir / "config.json"
        
        try:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            return config_path
        except Exception as e:
            raise IOError(f"Failed to save config to {config_path}") from e
    
    def load_config(self, experiment_name: str, version: str) -> Dict[str, Any]:
        """Load experiment configuration.
        
        Args:
            experiment_name: Name of the experiment.
            version: Version identifier.
        
        Returns:
            Dictionary of configuration.
        
        Raises:
            FileNotFoundError: If config file doesn't exist.
            IOError: If config cannot be loaded.
        """
        experiment_dir = self._get_experiment_dir(experiment_name, version)
        config_path = experiment_dir / "config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            return config
        except Exception as e:
            raise IOError(f"Failed to load config from {config_path}") from e
    
    def list_experiments(self) -> list[str]:
        """List all experiment names.
        
        Returns:
            List of experiment names.
        """
        if not self.base_dir.exists():
            return []
        
        experiments = [
            d.name for d in self.base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]
        return sorted(experiments)
    
    def list_versions(self, experiment_name: str) -> list[str]:
        """List all versions for an experiment.
        
        Args:
            experiment_name: Name of the experiment.
        
        Returns:
            List of version identifiers.
        """
        experiment_path = self.base_dir / experiment_name
        
        if not experiment_path.exists():
            return []
        
        versions = [
            d.name for d in experiment_path.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]
        return sorted(versions)

