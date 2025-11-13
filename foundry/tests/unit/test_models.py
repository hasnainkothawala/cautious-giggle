"""Unit tests for model components."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from ml_framework.models.base_model import BaseModel
from ml_framework.models.auto import AutoModel
from ml_framework.models.training_arguments import TrainingArguments
from ml_framework.models.trainer import ModelTrainer, TrainingResult
from ml_framework.models.wrappers.sklearn_wrapper import SklearnModelWrapper


class TestBaseModel:
    """Tests for BaseModel abstract class."""
    
    def test_base_model_is_abstract(self) -> None:
        """Test that BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel()  # type: ignore
    
    def test_base_model_has_required_methods(self) -> None:
        """Test that BaseModel defines required methods."""
        assert hasattr(BaseModel, "train")
        assert hasattr(BaseModel, "predict")
        assert hasattr(BaseModel, "save")
        assert hasattr(BaseModel, "load")
        assert BaseModel.train.__isabstractmethod__  # type: ignore


class TestSklearnModelWrapper:
    """Tests for SklearnModelWrapper."""
    
    def test_sklearn_wrapper_initialization(self) -> None:
        """Test SklearnModelWrapper can be initialized."""
        model = SklearnModelWrapper(model_type="logistic_regression")
        assert model.model_type == "logistic_regression"
        assert model.task == "classification"
        assert not model.trained
    
    def test_sklearn_wrapper_logistic_regression(self, sample_training_data) -> None:
        """Test SklearnModelWrapper with LogisticRegression."""
        X_train, y_train = sample_training_data
        model = SklearnModelWrapper(model_type="logistic_regression")
        
        model.train(X_train, y_train)
        
        assert model.trained
        predictions = model.predict(X_train)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y_train)
    
    def test_sklearn_wrapper_random_forest_classification(self, sample_training_data) -> None:
        """Test SklearnModelWrapper with RandomForestClassifier."""
        X_train, y_train = sample_training_data
        model = SklearnModelWrapper(model_type="random_forest", task="classification")
        
        model.train(X_train, y_train)
        
        assert model.trained
        predictions = model.predict(X_train)
        assert isinstance(predictions, np.ndarray)
    
    def test_sklearn_wrapper_predict_proba(self, sample_training_data) -> None:
        """Test SklearnModelWrapper predict_proba method."""
        X_train, y_train = sample_training_data
        model = SklearnModelWrapper(model_type="logistic_regression")
        model.train(X_train, y_train)
        
        probabilities = model.predict_proba(X_train)
        
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape[0] == len(X_train)
        # Check probabilities sum to 1
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_sklearn_wrapper_save_load(self, sample_training_data, tmp_path: Path) -> None:
        """Test SklearnModelWrapper save and load."""
        X_train, y_train = sample_training_data
        model = SklearnModelWrapper(model_type="logistic_regression")
        model.train(X_train, y_train)
        
        # Save model
        model_path = tmp_path / "model.pkl"
        model.save(model_path)
        assert model_path.exists()
        
        # Load model
        new_model = SklearnModelWrapper(model_type="logistic_regression")
        new_model.load(model_path)
        
        assert new_model.trained
        predictions = new_model.predict(X_train)
        assert len(predictions) == len(y_train)
    
    def test_sklearn_wrapper_train_before_predict(self, sample_training_data) -> None:
        """Test SklearnModelWrapper requires training before prediction."""
        X_train, y_train = sample_training_data
        model = SklearnModelWrapper(model_type="logistic_regression")
        
        with pytest.raises(ValueError, match="must be trained"):
            model.predict(X_train)
    
    def test_sklearn_wrapper_empty_data(self) -> None:
        """Test SklearnModelWrapper handles empty data."""
        model = SklearnModelWrapper(model_type="logistic_regression")
        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype=int)
        
        with pytest.raises(ValueError, match="empty"):
            model.train(empty_df, empty_series)
    
    def test_sklearn_wrapper_invalid_model_type(self) -> None:
        """Test SklearnModelWrapper raises error for invalid model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            SklearnModelWrapper(model_type="invalid_model")
    
    def test_sklearn_wrapper_invalid_task(self) -> None:
        """Test SklearnModelWrapper raises error for invalid task."""
        with pytest.raises(ValueError, match="only supports classification"):
            SklearnModelWrapper(model_type="logistic_regression", task="regression")


class TestAutoModel:
    """Tests for AutoModel factory."""
    
    def test_auto_model_logistic_regression(self) -> None:
        """Test AutoModel creates LogisticRegression."""
        model = AutoModel.from_type("logistic_regression")
        
        assert isinstance(model, SklearnModelWrapper)
        assert model.model_type == "logistic_regression"
    
    def test_auto_model_random_forest(self) -> None:
        """Test AutoModel creates RandomForest."""
        model = AutoModel.from_type("random_forest", task="classification")
        
        assert isinstance(model, SklearnModelWrapper)
        assert model.model_type == "random_forest"
    
    def test_auto_model_with_kwargs(self) -> None:
        """Test AutoModel passes kwargs to model."""
        model = AutoModel.from_type("random_forest", task="classification", n_estimators=50)
        
        assert isinstance(model, SklearnModelWrapper)
        # Check that kwargs were passed (model should have n_estimators=50)
        assert hasattr(model.model, "n_estimators")
    
    def test_auto_model_unknown_type(self) -> None:
        """Test AutoModel raises error for unknown model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            AutoModel.from_type("unknown_model")
    
    def test_auto_model_register_model(self) -> None:
        """Test AutoModel can register new models."""
        # Create a simple test model
        class TestModel(BaseModel):
            def train(self, X: pd.DataFrame, y: pd.Series) -> None:
                self.trained = True
            
            def predict(self, X: pd.DataFrame) -> np.ndarray:
                return np.array([0] * len(X))
            
            def save(self, path: Path) -> None:
                pass
            
            def load(self, path: Path) -> None:
                pass
        
        # Register test model
        AutoModel.register_model("test_model", TestModel)
        
        # Use it
        model = AutoModel.from_type("test_model")
        assert isinstance(model, TestModel)


class TestTrainingArguments:
    """Tests for TrainingArguments."""
    
    def test_training_arguments_initialization(self) -> None:
        """Test TrainingArguments can be initialized."""
        args = TrainingArguments(output_dir="./experiments")
        
        assert args.output_dir == "./experiments"
        assert args.num_epochs == 100
        assert args.evaluation_strategy == "epoch"
    
    def test_training_arguments_custom_values(self) -> None:
        """Test TrainingArguments accepts custom values."""
        args = TrainingArguments(
            output_dir="./custom",
            num_epochs=50,
            evaluation_strategy="no",
            save_strategy="step",
            save_steps=100
        )
        
        assert args.output_dir == "./custom"
        assert args.num_epochs == 50
        assert args.evaluation_strategy == "no"
        assert args.save_strategy == "step"
        assert args.save_steps == 100
    
    def test_training_arguments_immutable(self) -> None:
        """Test TrainingArguments is immutable (frozen)."""
        args = TrainingArguments(output_dir="./experiments")
        
        with pytest.raises(Exception):  # Pydantic validation error
            args.num_epochs = 200  # type: ignore
    
    def test_training_arguments_extra_fields_forbidden(self) -> None:
        """Test TrainingArguments rejects extra fields."""
        with pytest.raises(Exception):  # Pydantic validation error
            TrainingArguments(output_dir="./experiments", invalid_field=123)  # type: ignore


class TestModelTrainer:
    """Tests for ModelTrainer."""
    
    def test_model_trainer_initialization(self, sample_training_data) -> None:
        """Test ModelTrainer can be initialized."""
        X_train, y_train = sample_training_data
        model = AutoModel.from_type("logistic_regression")
        args = TrainingArguments(output_dir="./experiments")
        
        trainer = ModelTrainer(
            model=model,
            args=args,
            train_data=(X_train, y_train)
        )
        
        assert trainer.model == model
        assert trainer.args == args
    
    def test_model_trainer_train(self, sample_training_data, tmp_path: Path) -> None:
        """Test ModelTrainer trains model."""
        X_train, y_train = sample_training_data
        model = AutoModel.from_type("logistic_regression")
        args = TrainingArguments(output_dir=str(tmp_path / "experiments"))
        
        trainer = ModelTrainer(
            model=model,
            args=args,
            train_data=(X_train, y_train),
            eval_data=(X_train, y_train)  # Use same data for eval
        )
        
        result = trainer.train()
        
        assert isinstance(result, TrainingResult)
        assert result.model.trained
        assert "accuracy" in result.metrics or "mse" in result.metrics
        assert result.version is not None
    
    def test_model_trainer_empty_data(self) -> None:
        """Test ModelTrainer handles empty training data."""
        model = AutoModel.from_type("logistic_regression")
        args = TrainingArguments(output_dir="./experiments")
        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype=int)
        
        trainer = ModelTrainer(
            model=model,
            args=args,
            train_data=(empty_df, empty_series)
        )
        
        with pytest.raises(ValueError, match="cannot be empty"):
            trainer.train()
    
    def test_model_trainer_mismatched_data_lengths(self) -> None:
        """Test ModelTrainer handles mismatched data lengths."""
        model = AutoModel.from_type("logistic_regression")
        args = TrainingArguments(output_dir="./experiments")
        X = pd.DataFrame({"feature": [1, 2, 3]})
        y = pd.Series([0, 1])  # Different length
        
        trainer = ModelTrainer(
            model=model,
            args=args,
            train_data=(X, y)
        )
        
        with pytest.raises(ValueError, match="same length"):
            trainer.train()

