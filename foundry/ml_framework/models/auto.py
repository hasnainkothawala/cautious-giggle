"""Auto model factory.

Provides easy instantiation of models using a factory pattern.
"""

from typing import Optional, Dict, Type, Any

from ml_framework.models.base_model import BaseModel
from ml_framework.models.wrappers.sklearn_wrapper import SklearnModelWrapper


class AutoModel:
    """Factory for creating models by type name."""
    
    _model_registry: Dict[str, Type[BaseModel]] = {
        "logistic_regression": SklearnModelWrapper,
        "random_forest": SklearnModelWrapper,
        # Future: Add more models here
        # "xgboost": XGBoostWrapper,
        # "neural_network": NeuralNetworkWrapper,
    }
    
    @classmethod
    def from_type(cls, model_type: str, **kwargs: Any) -> BaseModel:
        """Create model by type name.
        
        Args:
            model_type: Type of model ('logistic_regression', 'random_forest')
            **kwargs: Model parameters
        
        Returns:
            Model instance
        """
        if model_type not in cls._model_registry:
            available = ", ".join(cls._model_registry.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available types: {available}"
            )
        
        model_class = cls._model_registry[model_type]
        
        # For scikit-learn models, pass model_type to wrapper
        if model_class == SklearnModelWrapper:
            return model_class(model_type=model_type, **kwargs)
        
        # For other models, instantiate directly
        return model_class(**kwargs)
    
    @classmethod
    def register_model(cls, model_type: str, model_class: Type[BaseModel]) -> None:
        """Register a custom model type.
        
        Args:
            model_type: Model type name
            model_class: Model class
        """
        cls._model_registry[model_type] = model_class

