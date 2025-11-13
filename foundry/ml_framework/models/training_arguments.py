"""Training arguments configuration.

Defines configuration options for model training using Pydantic models.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class TrainingArguments(BaseModel):
    """Configuration for model training."""
    
    output_dir: str = Field(..., description="Directory to save outputs")
    num_epochs: int = Field(default=100, description="Number of training epochs")
    batch_size: Optional[int] = Field(default=None, description="Batch size for training")
    learning_rate: Optional[float] = Field(default=None, description="Learning rate")
    evaluation_strategy: Literal["no", "epoch", "step"] = Field(
        default="epoch", description="When to evaluate"
    )
    save_strategy: Literal["no", "epoch", "step"] = Field(
        default="epoch", description="When to save checkpoints"
    )
    logging_steps: int = Field(default=10, description="Number of steps between logging")
    save_steps: Optional[int] = Field(
        default=None, description="Number of steps between saves (if save_strategy='step')"
    )
    load_best_model_at_end: bool = Field(
        default=True, description="Whether to load best model at end of training"
    )
    metric_for_best_model: str = Field(
        default="f1_score", description="Metric name for selecting best model"
    )
    
    class Config:
        """Pydantic configuration."""
        frozen = True  # Immutable configuration
        extra = "forbid"  # Reject extra fields

