"""Model trainer class.

Orchestrates model training, evaluation, checkpointing, and logging.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from ml_framework.models.base_model import BaseModel
from ml_framework.models.training_arguments import TrainingArguments


class TrainingResult:
    """Container for training results."""
    
    def __init__(
        self,
        model: BaseModel,
        metrics: Dict[str, float],
        version: str,
        training_args: TrainingArguments,
    ) -> None:
        self.model = model
        self.metrics = metrics
        self.version = version
        self.training_args = training_args
        self.timestamp = datetime.now().isoformat()


def _generate_version() -> str:
    """Generate version string."""
    timestamp = datetime.now().strftime("%Y%m%d")
    unique_id = str(uuid.uuid4())[:8]
    return f"v1.0.0-{timestamp}-{unique_id}"


class ModelTrainer:
    """Handles model training with evaluation and checkpointing."""
    
    def __init__(
        self,
        model: BaseModel,
        args: TrainingArguments,
        train_data: Tuple[pd.DataFrame, pd.Series],
        eval_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> None:
        """Initialize model trainer.
        
        Args:
            model: Model instance implementing BaseModel interface.
            args: Training arguments configuration.
            train_data: Training data as (X, y) tuple.
            eval_data: Optional evaluation data as (X, y) tuple.
        """
        self.model = model
        self.args = args
        self.train_data = train_data
        self.eval_data = eval_data
        self.best_metrics: Optional[Dict[str, float]] = None
        self.best_model_path: Optional[Path] = None
    
    def train(self) -> TrainingResult:
        """Train model and return results.
        
        Handles training loop, evaluation, checkpointing, and logging.
        
        Returns:
            TrainingResult containing trained model, metrics, and version.
        
        Raises:
            ValueError: If training data is invalid.
        """
        X_train, y_train = self.train_data
        
        # Validate training data
        if X_train.empty or y_train.empty:
            raise ValueError("Training data cannot be empty")
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have same length")
        
        # Create output directory
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Train model
        self.model.train(X_train, y_train)
        
        # Evaluate if evaluation data provided
        metrics: Dict[str, float] = {}
        if self.eval_data is not None:
            metrics = self._evaluate()
            self.best_metrics = metrics.copy()
        
        # Save checkpoint if save_strategy is not "no"
        if self.args.save_strategy != "no":
            checkpoint_path = self._save_checkpoint(epoch=0)  # Single epoch for scikit-learn
            if checkpoint_path:
                self.best_model_path = checkpoint_path
        
        # Generate version identifier
        version = _generate_version()
        
        # Create training result
        result = TrainingResult(
            model=self.model,
            metrics=metrics,
            version=version,
            training_args=self.args,
        )
        
        return result
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate model on evaluation data.
        
        Returns:
            Dictionary of evaluation metrics.
        """
        if self.eval_data is None:
            return {}
        
        X_eval, y_eval = self.eval_data
        
        # Make predictions
        predictions = self.model.predict(X_eval)
        
        # Determine task type (classification vs regression)
        # Simple heuristic: if targets are integers and limited range, likely classification
        is_classification = (
            y_eval.dtype in [int, "int64", "int32"]
            and y_eval.nunique() < 20
        )
        
        # Calculate metrics
        metrics: Dict[str, float] = {}
        
        if is_classification:
            metrics["accuracy"] = float(accuracy_score(y_eval, predictions))
            metrics["precision"] = float(precision_score(y_eval, predictions, average="weighted", zero_division=0))
            metrics["recall"] = float(recall_score(y_eval, predictions, average="weighted", zero_division=0))
            metrics["f1_score"] = float(f1_score(y_eval, predictions, average="weighted", zero_division=0))
        else:
            metrics["mse"] = float(mean_squared_error(y_eval, predictions))
            metrics["mae"] = float(mean_absolute_error(y_eval, predictions))
            metrics["r2_score"] = float(r2_score(y_eval, predictions))
        
        return metrics
    
    def _save_checkpoint(self, epoch: int) -> Optional[Path]:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number.
        
        Returns:
            Path to saved checkpoint, or None if not saved.
        """
        output_dir = Path(self.args.output_dir)
        
        # Determine checkpoint path
        if self.args.save_strategy == "epoch":
            checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch}"
        elif self.args.save_strategy == "step":
            step = epoch * (self.args.save_steps or 1)
            checkpoint_dir = output_dir / f"checkpoint-step-{step}"
        else:
            return None
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_path = checkpoint_dir / "model.pkl"
        
        # Save model
        self.model.save(model_path)
        
        return model_path