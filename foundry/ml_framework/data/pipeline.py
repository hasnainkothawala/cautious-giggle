"""Composable data pipeline.

Enables chaining of data loaders and transformers for flexible data processing.
"""

from typing import List, Union, Optional
from pathlib import Path

import pandas as pd

from ml_framework.data.loaders.base_loader import BaseLoader
from ml_framework.data.transformers.base_transformer import BaseTransformer


class DataPipeline:
    """Chain loaders and transformers into a reusable pipeline."""
    
    def __init__(self, steps: List[Union[BaseLoader, BaseTransformer]]) -> None:
        """Initialize pipeline.
        
        Args:
            steps: Pipeline steps (must start with a loader)
        """
        if not steps:
            raise ValueError("Pipeline must have at least one step (a loader)")
        
        if not isinstance(steps[0], BaseLoader):
            raise ValueError("Pipeline must start with a data loader")
        
        self.steps = steps
    
    def execute(self, loader_path: Optional[Path] = None) -> pd.DataFrame:
        """Run all pipeline steps.
        
        Args:
            loader_path: Optional path for the loader
        
        Returns:
            Transformed DataFrame
        """
        data: Optional[pd.DataFrame] = None
        
        for i, step in enumerate(self.steps):
            if isinstance(step, BaseLoader):
                # Load data
                if i != 0:
                    raise ValueError(
                        "Data loaders can only be the first step in a pipeline"
                    )
                data = step.load(loader_path) if loader_path else step.load()
            
            elif isinstance(step, BaseTransformer):
                # Transform data
                if data is None:
                    raise RuntimeError(
                        "Cannot apply transformer before data is loaded. "
                        "Pipeline must start with a loader."
                    )
                
                # Fit transformer on first pass (if not already fitted)
                if not hasattr(step, 'fitted') or not step.fitted:
                    step.fit(data)
                
                data = step.transform(data)
            
            else:
                raise ValueError(
                    f"Unknown step type: {type(step)}. "
                    "Steps must be BaseLoader or BaseTransformer instances."
                )
        
        if data is None:
            raise RuntimeError("Pipeline execution failed - no data produced")
        
        return data
    
    def fit_transform(self, loader_path: Optional[Path] = None) -> pd.DataFrame:
        """Fit and transform (alias for execute).
        
        Args:
            loader_path: Optional path for the loader
        
        Returns:
            Transformed DataFrame
        """
        return self.execute(loader_path)

