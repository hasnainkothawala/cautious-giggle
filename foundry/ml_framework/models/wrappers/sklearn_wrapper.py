"""scikit-learn model wrapper.

Wraps scikit-learn models to provide consistent interface with BaseModel.
"""

from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ml_framework.models.base_model import BaseModel


class SklearnModelWrapper(BaseModel):
    """Wraps sklearn models with a consistent interface."""
    
    def __init__(
        self,
        model_type: str = "logistic_regression",
        task: str = "classification",
        **kwargs: Any,
    ) -> None:
        """Initialize wrapper.
        
        Args:
            model_type: Model type
            task: 'classification' or 'regression'
            **kwargs: Model parameters
        """
        super().__init__()
        self.model_type = model_type
        self.task = task
        self.model_kwargs = kwargs
        self.model: Optional[BaseEstimator] = None
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize sklearn model."""
        if self.model_type == "logistic_regression":
            if self.task != "classification":
                raise ValueError("LogisticRegression only supports classification")
            self.model = LogisticRegression(**self.model_kwargs)
        elif self.model_type == "random_forest":
            if self.task == "classification":
                self.model = RandomForestClassifier(**self.model_kwargs)
            elif self.task == "regression":
                self.model = RandomForestRegressor(**self.model_kwargs)
            else:
                raise ValueError(f"Invalid task for RandomForest: {self.task}")
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train model.
        
        Args:
            X: Features
            y: Targets
        """
        if X.empty:
            raise ValueError("Cannot train model on empty features")
        if y.empty:
            raise ValueError("Cannot train model on empty targets")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")
        
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Convert DataFrame/Series to numpy arrays (scikit-learn requirement)
        X_array = X.values
        y_array = y.values
        
        self.model.fit(X_array, y_array)
        self.trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict.
        
        Args:
            X: Features
        
        Returns:
            Predictions
        """
        if not self.trained or self.model is None:
            raise ValueError("Model must be trained before predict()")
        
        if X.empty:
            raise ValueError("Cannot predict on empty features")
        
        # Convert DataFrame to numpy array
        X_array = X.values
        
        predictions = self.model.predict(X_array)
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities (for classification models).
        
        Args:
            X: Features to predict on as DataFrame.
        
        Returns:
            Class probabilities as numpy array.
        
        Raises:
            ValueError: If model hasn't been trained.
            ValueError: If model doesn't support predict_proba.
        """
        if not self.trained or self.model is None:
            raise ValueError("Model must be trained before predict_proba()")
        
        if not hasattr(self.model, "predict_proba"):
            raise ValueError(f"Model {self.model_type} does not support predict_proba()")
        
        if X.empty:
            raise ValueError("Cannot predict on empty features")
        
        X_array = X.values
        probabilities = self.model.predict_proba(X_array)
        return probabilities
    
    def save(self, path: Path) -> None:
        """Save model.
        
        Args:
            path: Save path
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            joblib.dump(self.model, path)
        except Exception as e:
            raise IOError(f"Failed to save model to {path}") from e
    
    def load(self, path: Path) -> None:
        """Load model.
        
        Args:
            path: Load path
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            self.model = joblib.load(path)
            self.trained = True
        except Exception as e:
            raise IOError(f"Failed to load model from {path}") from e

