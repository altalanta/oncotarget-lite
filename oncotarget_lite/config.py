"""Configuration management system using Pydantic for validation and type safety.

This module provides centralized configuration for the oncotarget-lite pipeline
with environment-based overrides and comprehensive validation.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, Field, validator


class MLPConfig(BaseModel):
    """Configuration for MLP training parameters."""
    
    hidden_sizes: Tuple[int, ...] = Field(default=(32, 16), description="Hidden layer dimensions")
    dropout: float = Field(default=0.15, ge=0.0, le=1.0, description="Dropout probability")
    lr: float = Field(default=1e-3, gt=0.0, description="Learning rate for optimizer")
    epochs: int = Field(default=100, gt=0, description="Maximum training epochs")
    batch_size: int = Field(default=16, gt=0, description="Mini-batch size")
    patience: int = Field(default=10, gt=0, description="Early stopping patience")
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    
    @validator('hidden_sizes')
    def validate_hidden_sizes(cls, v):
        if not v or any(size <= 0 for size in v):
            raise ValueError("Hidden sizes must be positive integers")
        return v


class RandomForestConfig(BaseModel):
    """Configuration for Random Forest parameters."""
    
    n_estimators: int = Field(default=100, gt=0, description="Number of trees")
    max_depth: Optional[int] = Field(default=None, gt=0, description="Maximum tree depth")
    random_state: int = Field(default=42, description="Random seed for reproducibility")


class DataConfig(BaseModel):
    """Configuration for data loading and preprocessing."""
    
    data_dir: Optional[Path] = Field(default=None, description="Path to data directory")
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0, description="Test set proportion")
    val_size: float = Field(default=0.2, gt=0.0, lt=1.0, description="Validation set proportion")
    random_state: int = Field(default=42, description="Random seed for data splits")
    
    @validator('data_dir', pre=True)
    def resolve_data_dir(cls, v):
        if v is None:
            return Path(__file__).parent.parent / "data" / "raw"
        return Path(v)


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation parameters."""
    
    n_bootstrap: int = Field(default=1000, gt=0, description="Bootstrap samples for CIs")
    calibration_bins: int = Field(default=10, gt=0, description="Bins for calibration analysis")
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0, description="CI confidence level")
    random_state: int = Field(default=42, description="Random seed for bootstrap")


class MLFlowConfig(BaseModel):
    """Configuration for MLflow experiment tracking."""
    
    tracking_uri: str = Field(default="./mlruns", description="MLflow tracking URI")
    experiment_name: str = Field(default="oncotarget-lite", description="Experiment name")
    enable_tracking: bool = Field(default=True, description="Enable MLflow tracking")
    
    @validator('tracking_uri', pre=True)
    def get_tracking_uri(cls, v):
        return os.getenv("MLFLOW_TRACKING_URI", v)


class LoggingConfig(BaseModel):
    """Configuration for logging system."""
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    file_path: Optional[Path] = Field(default=None, description="Log file path")
    console_output: bool = Field(default=True, description="Enable console logging")
    
    @validator('level')
    def validate_level(cls, v):
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid logging level. Must be one of {valid_levels}")
        return v.upper()


class Config(BaseModel):
    """Main configuration object containing all sub-configurations."""
    
    mlp: MLPConfig = Field(default_factory=MLPConfig)
    random_forest: RandomForestConfig = Field(default_factory=RandomForestConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    mlflow: MLFlowConfig = Field(default_factory=MLFlowConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Global settings
    project_name: str = Field(default="oncotarget-lite", description="Project identifier")
    version: str = Field(default="0.2.0", description="Project version")
    
    class Config:
        """Pydantic configuration."""
        
        env_prefix = "ONCOTARGET_"
        case_sensitive = False
        validate_assignment = True
        extra = "forbid"


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from file with environment variable overrides.
    
    Args:
        config_path: Optional path to YAML/JSON configuration file
        
    Returns:
        Validated configuration object
        
    Example:
        >>> config = load_config()
        >>> config.mlp.hidden_sizes
        (32, 16)
    """
    if config_path and config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return Config(**config_data)
    
    # Load from environment variables with defaults
    return Config()


def get_output_dir(config: Config, model_type: str) -> Path:
    """Get output directory for model artifacts.
    
    Args:
        config: Configuration object
        model_type: Type of model (e.g., 'mlp', 'random_forest')
        
    Returns:
        Path to output directory
    """
    base_dir = Path("artifacts") / config.project_name / model_type
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir