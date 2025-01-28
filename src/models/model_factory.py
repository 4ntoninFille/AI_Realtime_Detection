from typing import Dict, Any, Optional

from .implementations.random_forest import RandomForest
from .implementations.xgboost import XGBoost

from .model_base import ModelBase

class ModelFactory:
    _models: Dict[str, type] = {
        'random_forest': RandomForest,
        'xgboost': XGBoost,
    }

    @classmethod
    def register_model(cls, model_type: str, model_class: type) -> None:
        """
        Register a new model type to the factory.
        
        Args:
            model_type: String
            model_class: The model class to register
        """
        if not issubclass(model_class, ModelBase):
            raise ValueError(f"Model class must inherit from ModelBase")
        cls._models[model_type] = model_class

    @classmethod
    def get_model(cls, 
                 model_type: str, 
                 config: Optional[Dict[str, Any]] = None, 
                 **kwargs) -> ModelBase:
        """
        Create and return a model instance of the specified type.
        
        Args:
            model_type: Type of model to create
            config: Configuration dictionary for the model
            **kwargs: Additional keyword arguments for model initialization
            
        Returns:
            An instance of the specified model
            
        Raises:
            ValueError: If model_type is not recognized
        """
        # Merge config and kwargs
        model_params = {**(config or {}), **kwargs}
        
        # Get the model class from registry
        if model_type not in cls._models:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available models: {list(cls._models.keys())}"
            )
        
        # Create and return the model instance
        model_class = cls._models[model_type]
        return model_class(**model_params)
