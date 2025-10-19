"""Prediction service for real-time inference."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..data import build_feature_matrix
from .model_loader import ModelLoader


@dataclass
class PredictionRequest:
    """Request for model prediction."""
    genes: List[str]
    model_name: str = "main"
    return_probabilities: bool = True
    include_explanations: bool = False


@dataclass
class PredictionResult:
    """Result of model prediction."""
    gene: str
    prediction: float
    probability: float
    model_name: str
    timestamp: float
    explanation: Optional[Dict[str, Any]] = None


@dataclass
class BatchPredictionResult:
    """Batch prediction results."""
    predictions: List[PredictionResult]
    model_name: str
    total_genes: int
    processing_time: float
    model_performance: Dict[str, float]


class PredictionService:
    """Service for making predictions with trained models."""

    def __init__(self, models_dir: Path = Path("models")):
        self.model_loader = ModelLoader(models_dir)
        self.prediction_cache: Dict[str, Any] = {}

    def predict_single(self, request: PredictionRequest) -> PredictionResult:
        """Make a single prediction."""

        # Load model
        model = self.model_loader.load_model(request.model_name)

        # For single gene prediction, we need to create a minimal feature matrix
        # In practice, this would be more sophisticated
        gene_features = self._prepare_gene_features([request.genes[0]])

        # Make prediction
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(gene_features)
            prediction = probabilities[0, 1]  # Probability of positive class
        else:
            prediction = float(model.predict(gene_features)[0])

        result = PredictionResult(
            gene=request.genes[0],
            prediction=prediction,
            probability=prediction if request.return_probabilities else 0.0,
            model_name=request.model_name,
            timestamp=time.time()
        )

        return result

    def predict_batch(self, request: PredictionRequest) -> BatchPredictionResult:
        """Make batch predictions."""

        start_time = time.time()

        # Load model
        model = self.model_loader.load_model(request.model_name)

        # Prepare features for all genes
        gene_features = self._prepare_gene_features(request.genes)

        # Make predictions
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(gene_features)
            predictions = probabilities[:, 1]  # Probabilities of positive class
        else:
            predictions = model.predict(gene_features).astype(float)

        # Create results
        results = []
        for i, gene in enumerate(request.genes):
            result = PredictionResult(
                gene=gene,
                prediction=predictions[i],
                probability=predictions[i] if request.return_probabilities else 0.0,
                model_name=request.model_name,
                timestamp=time.time()
            )
            results.append(result)

        # Get model performance metrics
        performance = self.model_loader.get_model_performance(request.model_name)

        processing_time = time.time() - start_time

        return BatchPredictionResult(
            predictions=results,
            model_name=request.model_name,
            total_genes=len(request.genes),
            processing_time=processing_time,
            model_performance=performance
        )

    def _prepare_gene_features(self, genes: List[str]) -> pd.DataFrame:
        """Prepare feature matrix for prediction."""

        # In a real deployment, this would:
        # 1. Query a feature store or database for gene features
        # 2. Apply the same preprocessing as during training
        # 3. Handle missing features gracefully

        # For now, we'll use a simplified approach
        # In practice, you'd want to replicate the exact feature engineering pipeline

        # Create a minimal feature matrix based on the training data structure
        # This is a placeholder - in production, you'd query real feature data

        # Get feature names from a reference model
        available_models = self.model_loader.list_available_models()
        if not available_models:
            # Return minimal features if no models available
            return pd.DataFrame(index=genes)

        # Use the first available model to get feature structure
        sample_model = self.model_loader.load_model(available_models[0]["name"])

        # Extract feature names from the pipeline
        if hasattr(sample_model, 'feature_names_in_'):
            feature_names = sample_model.feature_names_in_
        else:
            # Fallback: assume standard features
            feature_names = ['ppi_degree', 'signal_peptide', 'ig_like_domain', 'protein_length']

        # Create synthetic features (in production, this would be real data)
        features = pd.DataFrame(index=genes, columns=feature_names)

        # Fill with realistic synthetic values
        for col in feature_names:
            if col == 'ppi_degree':
                features[col] = np.random.poisson(5, size=len(genes))
            elif col in ['signal_peptide', 'ig_like_domain']:
                features[col] = np.random.choice([0, 1], size=len(genes))
            elif col == 'protein_length':
                features[col] = np.random.randint(100, 2000, size=len(genes))
            else:
                features[col] = np.random.normal(0, 1, size=len(genes))

        return features

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the prediction service."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "available_models": len(self.model_loader.list_available_models()),
            "cache_size": len(self.model_loader.model_cache)
        }

    def get_model_comparison(self) -> Dict[str, Any]:
        """Compare performance across available models."""
        models = self.model_loader.list_available_models()

        comparison = {
            "models": [],
            "best_model": None,
            "best_metric": 0
        }

        for model in models:
            perf = self.model_loader.get_model_performance(model["name"])
            if perf:
                model_info = {
                    "name": model["name"],
                    "type": model["type"],
                    "metrics": perf
                }
                comparison["models"].append(model_info)

                # Find best model by AUROC
                if "auroc" in perf and perf["auroc"] > comparison["best_metric"]:
                    comparison["best_metric"] = perf["auroc"]
                    comparison["best_model"] = model["name"]

        return comparison



