"""Reusable test fixtures factory for oncotarget-lite tests.

This module provides a centralized factory for generating test data,
reducing duplication across test files and ensuring consistency.

Usage:
    from tests.fixtures import TestDataFactory, create_synthetic_predictions

    # Create a factory instance
    factory = TestDataFactory(seed=42)

    # Generate test data
    predictions = factory.create_predictions(n_samples=100)
    features = factory.create_features(n_genes=50, n_features=10)

    # Or use convenience functions
    predictions = create_synthetic_predictions(n_train=40, n_test=20)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TestDataFactory:
    """Factory for generating consistent test data across all tests.

    This class centralizes test data generation to ensure:
    - Reproducibility via seeded random number generation
    - Consistency in data format across all tests
    - DRY principle - no duplicated data generation code
    """

    seed: int = 42
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def create_predictions(
        self,
        n_samples: int = 100,
        positive_ratio: float = 0.5,
        positive_mean: float = 0.8,
        negative_mean: float = 0.2,
        std: float = 0.1,
        split: str = "test",
        gene_prefix: str = "GENE",
    ) -> pd.DataFrame:
        """Create synthetic prediction data with realistic probability distributions.

        Args:
            n_samples: Number of samples to generate
            positive_ratio: Fraction of positive labels
            positive_mean: Mean probability for positive class predictions
            negative_mean: Mean probability for negative class predictions
            std: Standard deviation for probability distributions
            split: Split label (train/test)
            gene_prefix: Prefix for gene names

        Returns:
            DataFrame with columns: gene, split, y_prob, y_true
        """
        n_positive = int(n_samples * positive_ratio)
        n_negative = n_samples - n_positive

        # Generate labels
        labels = np.array([1] * n_positive + [0] * n_negative)
        self.rng.shuffle(labels)

        # Generate probabilities correlated with labels
        probs = np.where(
            labels == 1,
            self.rng.normal(positive_mean, std, size=n_samples),
            self.rng.normal(negative_mean, std, size=n_samples),
        )
        probs = np.clip(probs, 0.01, 0.99)

        genes = [f"{gene_prefix}_{i:04d}" for i in range(n_samples)]

        return pd.DataFrame({
            "gene": genes,
            "split": split,
            "y_prob": probs,
            "y_true": labels,
        })

    def create_train_test_predictions(
        self,
        n_train: int = 40,
        n_test: int = 20,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Create combined train/test prediction data.

        Args:
            n_train: Number of training samples
            n_test: Number of test samples
            **kwargs: Additional arguments passed to create_predictions

        Returns:
            Combined DataFrame with train and test predictions
        """
        train = self.create_predictions(
            n_samples=n_train,
            split="train",
            gene_prefix="train",
            **kwargs,
        )
        test = self.create_predictions(
            n_samples=n_test,
            split="test",
            gene_prefix="test",
            **kwargs,
        )
        return pd.concat([train, test], ignore_index=True)

    def create_features(
        self,
        n_genes: int = 100,
        n_features: int = 10,
        feature_names: list[str] | None = None,
        gene_names: list[str] | None = None,
        include_nan: bool = False,
        nan_ratio: float = 0.05,
    ) -> pd.DataFrame:
        """Create synthetic feature matrix.

        Args:
            n_genes: Number of genes (rows)
            n_features: Number of features (columns)
            feature_names: Custom feature names (auto-generated if None)
            gene_names: Custom gene names (auto-generated if None)
            include_nan: Whether to include NaN values
            nan_ratio: Fraction of NaN values if include_nan is True

        Returns:
            DataFrame with features indexed by gene names
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        if gene_names is None:
            gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]

        data = self.rng.standard_normal((n_genes, n_features))

        if include_nan:
            mask = self.rng.random((n_genes, n_features)) < nan_ratio
            data[mask] = np.nan

        return pd.DataFrame(
            data,
            columns=feature_names,
            index=gene_names,
        )

    def create_labels(
        self,
        n_genes: int = 100,
        positive_ratio: float = 0.3,
        gene_names: list[str] | None = None,
    ) -> pd.Series:
        """Create synthetic binary labels.

        Args:
            n_genes: Number of genes
            positive_ratio: Fraction of positive labels
            gene_names: Custom gene names (auto-generated if None)

        Returns:
            Series with binary labels indexed by gene names
        """
        if gene_names is None:
            gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]

        n_positive = int(n_genes * positive_ratio)
        labels = np.array([1] * n_positive + [0] * (n_genes - n_positive))
        self.rng.shuffle(labels)

        return pd.Series(labels, index=gene_names, name="label")

    def create_api_request(
        self,
        n_features: int = 10,
        feature_names: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Create a synthetic API prediction request.

        Args:
            n_features: Number of features
            feature_names: Custom feature names

        Returns:
            Dictionary suitable for API request body
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        features = {
            name: float(self.rng.standard_normal())
            for name in feature_names
        }

        return {"features": features}

    def create_shap_values(
        self,
        n_samples: int = 100,
        n_features: int = 10,
        feature_names: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Create synthetic SHAP values for testing.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            feature_names: Custom feature names

        Returns:
            Tuple of (SHAP values array, feature names)
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        shap_values = self.rng.standard_normal((n_samples, n_features)) * 0.1

        return shap_values, feature_names

    def create_model_metrics(
        self,
        auroc: float | None = None,
        ap: float | None = None,
        accuracy: float | None = None,
        f1: float | None = None,
    ) -> dict[str, float]:
        """Create realistic model performance metrics.

        Args:
            auroc: Override AUROC value
            ap: Override Average Precision
            accuracy: Override accuracy
            f1: Override F1 score

        Returns:
            Dictionary of metrics
        """
        return {
            "auroc": auroc or float(self.rng.uniform(0.7, 0.95)),
            "ap": ap or float(self.rng.uniform(0.6, 0.9)),
            "accuracy": accuracy or float(self.rng.uniform(0.75, 0.92)),
            "f1": f1 or float(self.rng.uniform(0.65, 0.88)),
            "brier": float(self.rng.uniform(0.05, 0.15)),
            "ece": float(self.rng.uniform(0.01, 0.08)),
        }

    def save_predictions_to_parquet(
        self,
        path: Path,
        n_train: int = 40,
        n_test: int = 20,
    ) -> Path:
        """Save synthetic predictions to a parquet file.

        Args:
            path: Output path for the parquet file
            n_train: Number of training samples
            n_test: Number of test samples

        Returns:
            Path to the created file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        predictions = self.create_train_test_predictions(n_train, n_test)
        predictions.to_parquet(path, index=False)
        return path


# Convenience functions for quick test data generation


def create_synthetic_predictions(
    n_train: int = 40,
    n_test: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic train/test predictions.

    Convenience function for common use case.
    """
    factory = TestDataFactory(seed=seed)
    return factory.create_train_test_predictions(n_train, n_test)


def create_synthetic_features(
    n_genes: int = 100,
    n_features: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic feature matrix.

    Convenience function for common use case.
    """
    factory = TestDataFactory(seed=seed)
    return factory.create_features(n_genes, n_features)


def create_synthetic_api_request(
    n_features: int = 10,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Create synthetic API request.

    Convenience function for common use case.
    """
    factory = TestDataFactory(seed=seed)
    return factory.create_api_request(n_features)


__all__ = [
    "TestDataFactory",
    "create_synthetic_predictions",
    "create_synthetic_features",
    "create_synthetic_api_request",
]

