"""Base trainer interface for ablation experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass
class TrainerConfig:
    """Configuration for training experiments."""
    name: str
    model_type: str
    model_params: Dict[str, Any]
    feature_type: str
    feature_includes: list[str] | None = None
    seed: int = 42
    test_size: float = 0.3


@dataclass
class TrainingResult:
    """Result from training an ablation experiment."""
    config: TrainerConfig
    pipeline: Pipeline
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    feature_names: list[str]
    dataset_hash: str | None = None


class BaseTrainer(ABC):
    """Base class for all model trainers."""
    
    def __init__(self, config: TrainerConfig):
        self.config = config
    
    @abstractmethod
    def create_pipeline(self) -> Pipeline:
        """Create the sklearn pipeline for this trainer."""
        pass
    
    def filter_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Filter features based on configuration."""
        if self.config.feature_type == "all_features":
            return features_df
        elif self.config.feature_type == "advanced_features":
            # For advanced features, we'll use the features as-is since they were already
            # computed in build_feature_matrix with use_advanced_features=True
            return features_df
        elif self.config.feature_type == "clinical_only":
            if self.config.feature_includes:
                clinical_cols = []
                for pattern in self.config.feature_includes:
                    if pattern.endswith("*"):
                        prefix = pattern[:-1]
                        clinical_cols.extend([col for col in features_df.columns if col.startswith(prefix)])
                    else:
                        if pattern in features_df.columns:
                            clinical_cols.append(pattern)
                return features_df[clinical_cols]
            return features_df.filter(regex="tissue_type|age_group|sample_quality|batch_id")
        elif self.config.feature_type == "network_only":
            if self.config.feature_includes:
                network_cols = []
                for pattern in self.config.feature_includes:
                    if pattern.endswith("*"):
                        prefix = pattern[:-1]
                        network_cols.extend([col for col in features_df.columns if col.startswith(prefix)])
                    else:
                        if pattern in features_df.columns:
                            network_cols.append(pattern)
                return features_df[network_cols]
            return features_df.filter(regex="ppi_degree|centrality|network")
        else:
            raise ValueError(f"Unknown feature type: {self.config.feature_type}")
    
    def train(
        self,
        processed_dir: Path,
        models_dir: Path,
        reports_dir: Path,
    ) -> TrainingResult:
        """Train the model and return results."""
        from sklearn.metrics import roc_auc_score, average_precision_score
        import json

        # Load data
        features_df = pd.read_parquet(processed_dir / "features.parquet")
        labels_df = pd.read_parquet(processed_dir / "labels.parquet")

        with open(processed_dir / "splits.json") as f:
            splits = json.load(f)

        # Filter features
        features_df = self.filter_features(features_df)
        
        # Create pipeline
        pipeline = self.create_pipeline()
        
        # Split data
        train_idx = splits["train"]
        test_idx = splits["test"]
        
        X_train = features_df.iloc[train_idx]
        y_train = labels_df.iloc[train_idx]["is_cell_surface"]
        X_test = features_df.iloc[test_idx]
        y_test = labels_df.iloc[test_idx]["is_cell_surface"]
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        train_proba = pipeline.predict_proba(X_train)[:, 1]
        test_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Metrics
        train_metrics = {
            "auroc": roc_auc_score(y_train, train_proba),
            "ap": average_precision_score(y_train, train_proba),
        }
        
        test_metrics = {
            "auroc": roc_auc_score(y_test, test_proba),
            "ap": average_precision_score(y_test, test_proba),
        }
        
        return TrainingResult(
            config=self.config,
            pipeline=pipeline,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            feature_names=list(features_df.columns),
        )