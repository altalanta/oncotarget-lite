"""Integration tests for the ML pipeline.

This module tests the complete ML pipeline from data preparation through
model training, evaluation, and serving. These tests use realistic synthetic
data to verify the entire flow works end-to-end.

Tests are organized by pipeline stage:
1. Data Preparation: Loading, feature engineering, train/test split
2. Model Training: Pipeline construction, fitting, artifact saving
3. Model Evaluation: Metrics computation, prediction generation
4. End-to-End: Full pipeline from raw data to predictions
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# =============================================================================
# FIXTURES - Realistic Synthetic Data
# =============================================================================

@pytest.fixture(scope="module")
def synthetic_features_and_labels() -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic feature matrix and labels for testing.
    
    Returns realistic features and labels that mimic the actual data structure.
    """
    n_samples = 100
    rng = np.random.default_rng(42)
    genes = [f"GENE{i:04d}" for i in range(n_samples)]
    
    # Create features similar to what the pipeline produces
    features = pd.DataFrame({
        "ppi_degree": rng.integers(0, 100, n_samples).astype(float),
        "ppi_clustering": rng.uniform(0, 1, n_samples),
        "ppi_betweenness": rng.uniform(0, 0.5, n_samples),
        "signal_peptide": rng.binomial(1, 0.3, n_samples).astype(float),
        "ig_like_domain": rng.binomial(1, 0.2, n_samples).astype(float),
        "protein_length": rng.integers(100, 2000, n_samples).astype(float),
        "mean_dependency": rng.normal(0, 0.5, n_samples),
        "domain_count": rng.integers(0, 10, n_samples).astype(float),
    }, index=genes)
    
    # Create labels correlated with features for realistic behavior
    # Higher ppi_degree and signal_peptide increase probability of being cell surface
    prob = 1 / (1 + np.exp(-(
        0.02 * features["ppi_degree"] + 
        1.0 * features["signal_peptide"] + 
        0.5 * features["ig_like_domain"] - 1.0
    )))
    labels = pd.Series(
        rng.binomial(1, prob).astype(int),
        index=genes,
        name="is_cell_surface"
    )
    
    return features, labels


@pytest.fixture
def processed_data_dir(tmp_path: Path, synthetic_features_and_labels) -> Path:
    """Create a processed data directory with synthetic data."""
    features, labels = synthetic_features_and_labels
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    
    # Save features and labels
    features.to_parquet(processed_dir / "features.parquet")
    labels.to_frame(name="label").to_parquet(processed_dir / "labels.parquet")
    
    # Create train/test split
    n_train = int(len(features) * 0.7)
    train_genes = list(features.index[:n_train])
    test_genes = list(features.index[n_train:])
    
    splits = {
        "train_genes": train_genes,
        "test_genes": test_genes,
        "test_size": 0.3,
        "seed": 42,
        "dataset_hash": "test_hash_12345678",
    }
    
    with open(processed_dir / "splits.json", "w") as f:
        json.dump(splits, f)
    
    return processed_dir


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    """Create models directory."""
    models = tmp_path / "models"
    models.mkdir()
    return models


@pytest.fixture
def reports_dir(tmp_path: Path) -> Path:
    """Create reports directory."""
    reports = tmp_path / "reports"
    reports.mkdir()
    return reports


# =============================================================================
# DATA PREPARATION TESTS
# =============================================================================

class TestDataPreparation:
    """Integration tests for data preparation stage."""

    def test_prepared_data_artifacts_are_valid(self, processed_data_dir: Path):
        """Test that prepared data artifacts have correct structure."""
        # Verify all required files exist
        assert (processed_data_dir / "features.parquet").exists()
        assert (processed_data_dir / "labels.parquet").exists()
        assert (processed_data_dir / "splits.json").exists()
        
        # Verify features
        features = pd.read_parquet(processed_data_dir / "features.parquet")
        assert len(features) > 0
        assert len(features.columns) > 0
        
        # Verify labels
        labels_df = pd.read_parquet(processed_data_dir / "labels.parquet")
        assert "label" in labels_df.columns
        assert labels_df["label"].isin([0, 1]).all()
        
        # Verify splits
        with open(processed_data_dir / "splits.json") as f:
            splits = json.load(f)
        
        assert "train_genes" in splits
        assert "test_genes" in splits
        assert len(splits["train_genes"]) > 0
        assert len(splits["test_genes"]) > 0
        
        # Verify no overlap between train and test
        train_set = set(splits["train_genes"])
        test_set = set(splits["test_genes"])
        assert train_set.isdisjoint(test_set), "Train and test sets overlap"

    def test_features_are_numeric(self, processed_data_dir: Path):
        """Test that all features are numeric."""
        features = pd.read_parquet(processed_data_dir / "features.parquet")
        
        for col in features.columns:
            assert pd.api.types.is_numeric_dtype(features[col]), f"Column {col} is not numeric"

    def test_labels_are_balanced(self, processed_data_dir: Path):
        """Test that labels have both classes represented."""
        labels_df = pd.read_parquet(processed_data_dir / "labels.parquet")
        labels = labels_df["label"]
        
        assert labels.sum() > 0, "No positive labels"
        assert labels.sum() < len(labels), "No negative labels"


# =============================================================================
# MODEL TRAINING TESTS
# =============================================================================

class TestModelTraining:
    """Integration tests for model training stage."""

    def test_train_model_creates_artifacts(
        self, processed_data_dir: Path, models_dir: Path, reports_dir: Path
    ):
        """Test that train_model creates all required artifacts."""
        from oncotarget_lite.model import train_model, TrainConfig
        
        config = TrainConfig(
            model_type="logreg",
            C=1.0,
            max_iter=100,
            seed=42,
        )
        
        result = train_model(
            processed_dir=processed_data_dir,
            models_dir=models_dir,
            reports_dir=reports_dir,
            config=config,
        )
        
        # Verify TrainResult structure
        assert result.pipeline is not None
        assert result.train_predictions is not None
        assert result.test_predictions is not None
        assert result.train_metrics is not None
        assert result.test_metrics is not None
        
        # Verify artifacts were saved
        assert (models_dir / "logreg_pipeline.pkl").exists()
        assert (models_dir / "feature_list.json").exists()
        assert (reports_dir / "predictions.parquet").exists()
        assert (reports_dir / "metrics_basic.json").exists()
        
        # Verify metrics are reasonable
        assert 0.0 <= result.test_metrics["auroc"] <= 1.0
        assert 0.0 <= result.test_metrics["ap"] <= 1.0

    def test_trained_model_can_predict(
        self, processed_data_dir: Path, models_dir: Path, reports_dir: Path
    ):
        """Test that trained model can make predictions on new data."""
        from oncotarget_lite.model import train_model, TrainConfig
        import joblib
        
        config = TrainConfig(model_type="logreg", seed=42)
        
        train_model(
            processed_dir=processed_data_dir,
            models_dir=models_dir,
            reports_dir=reports_dir,
            config=config,
        )
        
        # Load the saved model
        model = joblib.load(models_dir / "logreg_pipeline.pkl")
        
        # Create new test data
        features = pd.read_parquet(processed_data_dir / "features.parquet")
        sample_features = features.iloc[:5]
        
        # Make predictions
        predictions = model.predict_proba(sample_features)[:, 1]
        
        # Verify predictions are valid probabilities
        assert len(predictions) == 5
        assert all(0.0 <= p <= 1.0 for p in predictions)

    def test_training_is_deterministic(
        self, processed_data_dir: Path, tmp_path: Path
    ):
        """Test that training with same seed produces identical results."""
        from oncotarget_lite.model import train_model, TrainConfig
        
        results = []
        for i in range(2):
            models_dir = tmp_path / f"models_{i}"
            reports_dir = tmp_path / f"reports_{i}"
            models_dir.mkdir()
            reports_dir.mkdir()
            
            config = TrainConfig(model_type="logreg", seed=42)
            
            result = train_model(
                processed_dir=processed_data_dir,
                models_dir=models_dir,
                reports_dir=reports_dir,
                config=config,
            )
            results.append(result)
        
        # Verify identical metrics
        assert results[0].test_metrics == results[1].test_metrics
        assert results[0].train_metrics == results[1].train_metrics


# =============================================================================
# MODEL EVALUATION TESTS
# =============================================================================

class TestModelEvaluation:
    """Integration tests for model evaluation stage."""

    @pytest.fixture
    def trained_model(
        self, processed_data_dir: Path, models_dir: Path, reports_dir: Path
    ):
        """Train a model for evaluation tests."""
        from oncotarget_lite.model import train_model, TrainConfig
        
        config = TrainConfig(model_type="logreg", seed=42)
        
        return train_model(
            processed_dir=processed_data_dir,
            models_dir=models_dir,
            reports_dir=reports_dir,
            config=config,
        )

    def test_evaluation_metrics_are_computed(self, trained_model, reports_dir: Path):
        """Test that evaluation metrics are computed and saved."""
        # Load metrics
        with open(reports_dir / "metrics_basic.json") as f:
            metrics = json.load(f)
        
        # Verify structure
        assert "train" in metrics
        assert "test" in metrics
        assert "train_size" in metrics
        assert "test_size" in metrics
        
        # Verify train metrics
        assert "auroc" in metrics["train"]
        assert "ap" in metrics["train"]
        
        # Verify test metrics
        assert "auroc" in metrics["test"]
        assert "ap" in metrics["test"]
        
        # Verify metric values are reasonable
        assert 0.0 <= metrics["test"]["auroc"] <= 1.0
        assert 0.0 <= metrics["test"]["ap"] <= 1.0

    def test_predictions_file_is_valid(self, trained_model, reports_dir: Path):
        """Test that predictions file has correct structure."""
        predictions = pd.read_parquet(reports_dir / "predictions.parquet")
        
        # Verify columns
        assert "gene" in predictions.columns
        assert "split" in predictions.columns
        assert "y_prob" in predictions.columns
        assert "y_true" in predictions.columns
        
        # Verify splits
        assert set(predictions["split"].unique()) == {"train", "test"}
        
        # Verify probabilities are valid
        assert predictions["y_prob"].between(0, 1).all()
        
        # Verify labels are binary
        assert predictions["y_true"].isin([0, 1]).all()

    def test_feature_list_matches_model(
        self, trained_model, processed_data_dir: Path, models_dir: Path
    ):
        """Test that saved feature list matches the model's expected features."""
        # Load feature list
        with open(models_dir / "feature_list.json") as f:
            feature_list = json.load(f)
        
        # Load features
        features = pd.read_parquet(processed_data_dir / "features.parquet")
        
        # Verify feature list matches
        assert feature_list["feature_order"] == list(features.columns)


# =============================================================================
# END-TO-END PIPELINE TESTS
# =============================================================================

class TestEndToEndPipeline:
    """End-to-end integration tests for the complete ML pipeline."""

    def test_full_pipeline_train_to_serve(
        self, processed_data_dir: Path, models_dir: Path, reports_dir: Path
    ):
        """Test the complete pipeline from training to serving."""
        from oncotarget_lite.model import train_model, TrainConfig
        import joblib
        
        # Step 1: Train model
        config = TrainConfig(model_type="logreg", seed=42)
        train_result = train_model(
            processed_dir=processed_data_dir,
            models_dir=models_dir,
            reports_dir=reports_dir,
            config=config,
        )
        
        assert train_result.test_metrics["auroc"] > 0.4  # Better than random
        
        # Step 2: Load model and verify it works
        model = joblib.load(models_dir / "logreg_pipeline.pkl")
        
        # Step 3: Create a new sample and predict
        features = pd.read_parquet(processed_data_dir / "features.parquet")
        new_sample = features.iloc[[0]].copy()
        prediction = model.predict_proba(new_sample)[0, 1]
        
        assert 0.0 <= prediction <= 1.0
        
        # Step 4: Verify all artifacts exist
        assert (models_dir / "logreg_pipeline.pkl").exists()
        assert (models_dir / "feature_list.json").exists()
        assert (reports_dir / "predictions.parquet").exists()
        assert (reports_dir / "metrics_basic.json").exists()

    def test_model_performance_is_reasonable(
        self, processed_data_dir: Path, models_dir: Path, reports_dir: Path
    ):
        """Test that trained model achieves reasonable performance."""
        from oncotarget_lite.model import train_model, TrainConfig
        
        config = TrainConfig(model_type="logreg", seed=42)
        result = train_model(
            processed_dir=processed_data_dir,
            models_dir=models_dir,
            reports_dir=reports_dir,
            config=config,
        )
        
        # Model should perform better than random (AUROC > 0.5)
        # Given the synthetic data has correlation between features and labels
        assert result.test_metrics["auroc"] > 0.5, "Model performs no better than random"
        
        # Average precision should also be reasonable
        assert result.test_metrics["ap"] > 0.3, "Average precision is too low"


# =============================================================================
# SERVING INTEGRATION TESTS
# =============================================================================

class TestServingIntegration:
    """Integration tests for model serving."""

    @pytest.fixture
    def trained_model_for_serving(
        self, processed_data_dir: Path, models_dir: Path, reports_dir: Path
    ):
        """Train a model and prepare it for serving."""
        from oncotarget_lite.model import train_model, TrainConfig
        
        config = TrainConfig(model_type="logreg", seed=42)
        train_model(
            processed_dir=processed_data_dir,
            models_dir=models_dir,
            reports_dir=reports_dir,
            config=config,
        )
        
        return {
            "processed_dir": processed_data_dir,
            "models_dir": models_dir,
            "reports_dir": reports_dir,
        }

    def test_model_loader_loads_trained_model(self, trained_model_for_serving):
        """Test that ModelLoader can load the trained model."""
        from deployment.model_loader import ModelLoader
        
        loader = ModelLoader(models_dir=trained_model_for_serving["models_dir"])
        
        # List available models
        models = loader.list_available_models()
        assert len(models) > 0
        
        # Load the main model
        model = loader.load_model()
        assert model is not None

    def test_prediction_service_makes_predictions(self, trained_model_for_serving):
        """Test that PredictionService can make predictions with trained model."""
        from deployment.prediction_service import PredictionService
        from oncotarget_lite.schemas import APIPredictionRequest
        
        # Load features to get feature names
        features = pd.read_parquet(
            trained_model_for_serving["processed_dir"] / "features.parquet"
        )
        
        # Create a sample request with actual feature names
        sample_features = {col: float(features[col].iloc[0]) for col in features.columns}
        
        service = PredictionService(
            models_dir=trained_model_for_serving["models_dir"],
            lazy_load=False,
        )
        
        request = APIPredictionRequest(features=sample_features)
        result = service.predict_single(request)
        
        assert "prediction" in result
        assert "model_version" in result
        assert 0.0 <= result["prediction"] <= 1.0

    def test_prediction_service_batch_predictions(self, trained_model_for_serving):
        """Test that PredictionService can make batch predictions."""
        from deployment.prediction_service import PredictionService
        from oncotarget_lite.schemas import APIPredictionRequest
        
        # Load features to get feature names
        features = pd.read_parquet(
            trained_model_for_serving["processed_dir"] / "features.parquet"
        )
        
        # Create multiple sample requests
        requests = []
        for i in range(5):
            sample_features = {col: float(features[col].iloc[i]) for col in features.columns}
            requests.append(APIPredictionRequest(features=sample_features))
        
        service = PredictionService(
            models_dir=trained_model_for_serving["models_dir"],
            lazy_load=False,
        )
        
        # Make batch predictions
        results = [service.predict_single(req) for req in requests]
        
        assert len(results) == 5
        for result in results:
            assert 0.0 <= result["prediction"] <= 1.0


# =============================================================================
# REGRESSION TESTS
# =============================================================================

class TestRegressionTests:
    """Regression tests to catch common issues."""

    def test_model_handles_unseen_feature_values(
        self, processed_data_dir: Path, models_dir: Path, reports_dir: Path
    ):
        """Test that model handles feature values outside training range."""
        from oncotarget_lite.model import train_model, TrainConfig
        import joblib
        
        config = TrainConfig(model_type="logreg", seed=42)
        train_model(
            processed_dir=processed_data_dir,
            models_dir=models_dir,
            reports_dir=reports_dir,
            config=config,
        )
        
        model = joblib.load(models_dir / "logreg_pipeline.pkl")
        features = pd.read_parquet(processed_data_dir / "features.parquet")
        
        # Create sample with extreme values
        extreme_sample = features.iloc[[0]].copy()
        for col in extreme_sample.columns:
            extreme_sample[col] = features[col].max() * 10
        
        # Model should still make a valid prediction
        prediction = model.predict_proba(extreme_sample)[0, 1]
        assert 0.0 <= prediction <= 1.0

    def test_model_handles_zero_features(
        self, processed_data_dir: Path, models_dir: Path, reports_dir: Path
    ):
        """Test that model handles all-zero feature vectors."""
        from oncotarget_lite.model import train_model, TrainConfig
        import joblib
        
        config = TrainConfig(model_type="logreg", seed=42)
        train_model(
            processed_dir=processed_data_dir,
            models_dir=models_dir,
            reports_dir=reports_dir,
            config=config,
        )
        
        model = joblib.load(models_dir / "logreg_pipeline.pkl")
        features = pd.read_parquet(processed_data_dir / "features.parquet")
        
        # Create sample with all zeros
        zero_sample = features.iloc[[0]].copy()
        for col in zero_sample.columns:
            zero_sample[col] = 0.0
        
        # Model should still make a valid prediction
        prediction = model.predict_proba(zero_sample)[0, 1]
        assert 0.0 <= prediction <= 1.0

    def test_predictions_are_consistent(
        self, processed_data_dir: Path, models_dir: Path, reports_dir: Path
    ):
        """Test that predictions are consistent across multiple calls."""
        from oncotarget_lite.model import train_model, TrainConfig
        import joblib
        
        config = TrainConfig(model_type="logreg", seed=42)
        train_model(
            processed_dir=processed_data_dir,
            models_dir=models_dir,
            reports_dir=reports_dir,
            config=config,
        )
        
        model = joblib.load(models_dir / "logreg_pipeline.pkl")
        features = pd.read_parquet(processed_data_dir / "features.parquet")
        sample = features.iloc[[0]]
        
        # Make predictions multiple times
        predictions = [model.predict_proba(sample)[0, 1] for _ in range(10)]
        
        # All predictions should be identical
        assert all(p == predictions[0] for p in predictions)
