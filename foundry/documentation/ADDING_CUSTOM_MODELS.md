# Adding Custom Models to the Framework

This guide shows how to add new model types that are **not** part of scikit-learn (e.g., XGBoost, PyTorch, TensorFlow, or custom models).

---

##  Overview

To add a custom model, you need to:

1. **Create a wrapper class** that implements `BaseModel` interface
2. **Register the model** with `AutoModel` factory
3. **Use it** just like any other model

---

##  Step-by-Step Guide

### Step 1: Create a Wrapper Class

Create a new file (e.g., `ml_framework/models/wrappers/xgboost_wrapper.py`) that implements `BaseModel`:

```python
"""XGBoost model wrapper.

Wraps XGBoost models to provide consistent interface with BaseModel.
"""

from pathlib import Path
from typing import Optional, Any

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb  # External library

from ml_framework.models.base_model import BaseModel


class XGBoostWrapper(BaseModel):
    """Wrapper for XGBoost models.
    
    Provides a consistent interface for XGBoost models by wrapping
    them in the BaseModel interface.
    
    **Example**:
        ```python
        model = XGBoostWrapper(task="classification", n_estimators=100)
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        ```
    """
    
    def __init__(
        self,
        task: str = "classification",
        **kwargs: Any,
    ) -> None:
        """Initialize XGBoost model wrapper.
        
        Args:
            task: Task type ("classification" or "regression").
            **kwargs: Additional arguments passed to XGBoost model constructor.
                Examples: n_estimators, max_depth, learning_rate, etc.
        """
        super().__init__()
        self.task = task
        self.model_kwargs = kwargs
        self.model: Optional[xgb.XGBModel] = None
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize XGBoost model based on task."""
        if self.task == "classification":
            self.model = xgb.XGBClassifier(**self.model_kwargs)
        elif self.task == "regression":
            self.model = xgb.XGBRegressor(**self.model_kwargs)
        else:
            raise ValueError(f"Invalid task: {self.task}. Must be 'classification' or 'regression'")
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the XGBoost model.
        
        Args:
            X: Training features as DataFrame.
            y: Training targets as Series.
        
        Raises:
            ValueError: If data is invalid or empty.
        """
        if X.empty:
            raise ValueError("Cannot train model on empty features")
        if y.empty:
            raise ValueError("Cannot train model on empty targets")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")
        
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Convert DataFrame/Series to numpy arrays (XGBoost requirement)
        X_array = X.values
        y_array = y.values
        
        self.model.fit(X_array, y_array)
        self.trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained model.
        
        Args:
            X: Features to predict on as DataFrame.
        
        Returns:
            Predictions as numpy array.
        
        Raises:
            ValueError: If model hasn't been trained.
            ValueError: If data is invalid or empty.
        """
        if not self.trained or self.model is None:
            raise ValueError("Model must be trained before predict()")
        
        if X.empty:
            raise ValueError("Cannot predict on empty features")
        
        # Convert DataFrame to numpy array
        X_array = X.values
        
        predictions = self.model.predict(X_array)
        return predictions
    
    def save(self, path: Path) -> None:
        """Save model to disk using joblib.
        
        Args:
            path: Path to save model file.
        
        Raises:
            IOError: If model cannot be saved.
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
        """Load model from disk using joblib.
        
        Args:
            path: Path to load model file from.
        
        Raises:
            FileNotFoundError: If model file doesn't exist.
            IOError: If model cannot be loaded.
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            self.model = joblib.load(path)
            self.trained = True
        except Exception as e:
            raise IOError(f"Failed to load model from {path}") from e
```

### Step 2: Register the Model

Update `ml_framework/models/auto.py` to register your new model:

```python
from ml_framework.models.wrappers.xgboost_wrapper import XGBoostWrapper

class AutoModel:
    _model_registry: Dict[str, Type[BaseModel]] = {
        "logistic_regression": SklearnModelWrapper,
        "random_forest": SklearnModelWrapper,
        "xgboost": XGBoostWrapper,  # Add your new model here
    }
    
    # ... rest of the code ...
```

**OR** register it dynamically (without modifying framework code):

```python
from ml_framework.models import AutoModel
from ml_framework.models.wrappers.xgboost_wrapper import XGBoostWrapper

# Register your custom model
AutoModel.register_model("xgboost", XGBoostWrapper)
```

### Step 3: Use Your Custom Model

Now you can use it just like any other model:

```python
from ml_framework.models import AutoModel, ModelTrainer, TrainingArguments

# Use XGBoost model
model = AutoModel.from_type(
    "xgboost",
    task="classification",
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)

training_args = TrainingArguments(output_dir="./experiments")
trainer = ModelTrainer(
    model=model,
    args=training_args,
    train_data=(X_train, y_train),
    eval_data=(X_test, y_test),
)

result = trainer.train()
```

---

## 🎨 Example: Custom Neural Network (PyTorch)

Here's how to add a PyTorch neural network:

```python
"""PyTorch neural network wrapper."""

from pathlib import Path
from typing import Optional, Any

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ml_framework.models.base_model import BaseModel


class PyTorchNNWrapper(BaseModel):
    """Wrapper for PyTorch neural networks."""
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] = [64, 32],
        output_size: int = 1,
        learning_rate: float = 0.001,
        **kwargs: Any,
    ) -> None:
        """Initialize PyTorch neural network.
        
        Args:
            input_size: Number of input features.
            hidden_sizes: List of hidden layer sizes.
            output_size: Number of output classes (1 for binary, N for multi-class).
            learning_rate: Learning rate for optimizer.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Build network
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss() if output_size == 1 else nn.CrossEntropyLoss()
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the neural network."""
        if X.empty or y.empty:
            raise ValueError("Cannot train on empty data")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.FloatTensor(y.values).unsqueeze(1) if self.output_size == 1 else torch.LongTensor(y.values)
        
        # Training loop
        self.model.train()
        for epoch in range(100):  # Simple training loop
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
        
        self.trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.trained:
            raise ValueError("Model must be trained before predict()")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values)
            outputs = self.model(X_tensor)
            
            if self.output_size == 1:
                predictions = (torch.sigmoid(outputs) > 0.5).int().numpy()
            else:
                predictions = torch.argmax(outputs, dim=1).numpy()
        
        return predictions
    
    def save(self, path: Path) -> None:
        """Save model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
        }, path)
    
    def load(self, path: Path) -> None:
        """Load model."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.trained = True
```

Then register it:

```python
AutoModel.register_model("neural_network", PyTorchNNWrapper)

# Use it
model = AutoModel.from_type(
    "neural_network",
    input_size=10,
    hidden_sizes=[64, 32],
    output_size=1
)
```

---

## 🎨 Example: Completely Custom Model

Here's a simple custom model (not using any external ML library):

```python
"""Custom simple linear model."""

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import joblib

from ml_framework.models.base_model import BaseModel


class CustomLinearModel(BaseModel):
    """Simple custom linear model: y = X @ weights + bias."""
    
    def __init__(self, learning_rate: float = 0.01) -> None:
        """Initialize custom model.
        
        Args:
            learning_rate: Learning rate for gradient descent.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train using simple gradient descent."""
        if X.empty or y.empty:
            raise ValueError("Cannot train on empty data")
        
        X_array = X.values
        y_array = y.values
        
        n_features = X_array.shape[1]
        self.weights = np.random.randn(n_features)
        self.bias = 0.0
        
        # Simple gradient descent
        for _ in range(1000):
            predictions = X_array @ self.weights + self.bias
            error = predictions - y_array
            
            # Update weights and bias
            self.weights -= self.learning_rate * (X_array.T @ error) / len(X_array)
            self.bias -= self.learning_rate * error.mean()
        
        self.trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.trained or self.weights is None:
            raise ValueError("Model must be trained before predict()")
        
        X_array = X.values
        predictions = X_array @ self.weights + self.bias
        return predictions
    
    def save(self, path: Path) -> None:
        """Save model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'weights': self.weights,
            'bias': self.bias,
        }, path)
    
    def load(self, path: Path) -> None:
        """Load model."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        data = joblib.load(path)
        self.weights = data['weights']
        self.bias = data['bias']
        self.trained = True
```

Register and use:

```python
AutoModel.register_model("custom_linear", CustomLinearModel)

model = AutoModel.from_type("custom_linear", learning_rate=0.01)
```

---

## ✅ Key Requirements

Your custom model class **must** implement all `BaseModel` methods:

1. **`train(X, y)`** - Train the model
2. **`predict(X)`** - Make predictions
3. **`save(path)`** - Save model to disk
4. **`load(path)`** - Load model from disk

---

## 🎯 Summary

**To add a custom model:**

1. ✅ Create a class that inherits from `BaseModel`
2. ✅ Implement all required methods (`train`, `predict`, `save`, `load`)
3. ✅ Register it with `AutoModel.register_model("model_name", YourModelClass)`
4. ✅ Use it: `AutoModel.from_type("model_name", **kwargs)`

**That's it!** The framework handles the rest (training orchestration, artifact saving, etc.).

