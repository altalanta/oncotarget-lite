"""Comprehensive tests for model.py to boost coverage."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import json
import joblib
from sklearn.pipeline import Pipeline

from oncotarget_lite.model import (
    TrainConfig, TrainResult, TrainingError, 
    _load_processed, _build_pipeline, _collect_predictions, 
    _compute_metrics, train_model
)
from oncotarget_lite.utils import save_json, save_dataframe


def create_test_processed_data(processed_dir: Path):
    """Create test processed data for model training."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create features
    features = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature2": [0.5, 1.5, 2.5, 3.5, 4.5],
        "feature3": [10.0, 20.0, 30.0, 40.0, 50.0],
    }, index=["GENE001", "GENE002", "GENE003", "GENE004", "GENE005"])
    
    # Create labels
    labels = pd.DataFrame({
        "label": [0, 1, 0, 1, 0]
    }, index=["GENE001", "GENE002", "GENE003", "GENE004", "GENE005"])
    
    # Create splits
    splits = {
        "train_genes": ["GENE001", "GENE002", "GENE003"],
        "test_genes": ["GENE004", "GENE005"],
        "dataset_hash": "test_hash_123"
    }
    
    # Save files
    features.to_parquet(processed_dir / "features.parquet")
    labels.to_parquet(processed_dir / "labels.parquet")
    save_json(processed_dir / "splits.json", splits)
    
    return features, labels, splits


def test_train_config_defaults():
    """Test TrainConfig default values."""
    config = TrainConfig()
    assert config.C == 1.0
    assert config.penalty == "l2"
    assert config.max_iter == 500
    assert config.class_weight == "balanced"
    assert config.seed == 42


def test_train_config_custom():
    """Test TrainConfig with custom values."""
    config = TrainConfig(
        C=0.5,
        penalty="l1",
        max_iter=1000,
        class_weight=None,
        seed=123
    )
    assert config.C == 0.5
    assert config.penalty == "l1"
    assert config.max_iter == 1000
    assert config.class_weight is None
    assert config.seed == 123


def test_training_error():
    """Test TrainingError exception."""
    error = TrainingError("Test training error")
    assert str(error) == "Test training error"
    assert isinstance(error, RuntimeError)


def test_load_processed_missing_files(tmp_path):
    """Test _load_processed with missing files."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    
    with pytest.raises(TrainingError) as exc_info:
        _load_processed(processed_dir)
    
    assert "Processed features/labels/splits not found" in str(exc_info.value)


def test_load_processed_success(tmp_path):
    """Test _load_processed with valid files."""
    processed_dir = tmp_path / "processed"
    features, labels, splits = create_test_processed_data(processed_dir)
    
    loaded_features, loaded_labels, loaded_splits = _load_processed(processed_dir)
    
    assert loaded_features.equals(features)
    assert loaded_labels.equals(labels["label"].astype(int))
    assert loaded_splits == splits


def test_build_pipeline_default():
    """Test _build_pipeline with default config."""
    config = TrainConfig()
    pipeline = _build_pipeline(config)
    
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == "scaler"
    assert pipeline.steps[1][0] == "clf"
    
    # Check LogisticRegression parameters
    clf = pipeline.steps[1][1]
    assert clf.C == 1.0
    assert clf.penalty == "l2"
    assert clf.max_iter == 500
    assert clf.class_weight == "balanced"
    assert clf.solver == "lbfgs"
    assert clf.random_state == 42


def test_build_pipeline_custom():
    """Test _build_pipeline with custom config."""
    config = TrainConfig(C=0.1, penalty="l1", max_iter=200, seed=999)
    pipeline = _build_pipeline(config)
    
    clf = pipeline.steps[1][1]
    assert clf.C == 0.1
    assert clf.penalty == "l1"
    assert clf.max_iter == 200
    assert clf.random_state == 999


def test_collect_predictions(tmp_path):
    """Test _collect_predictions function."""
    processed_dir = tmp_path / "processed"
    features, labels, splits = create_test_processed_data(processed_dir)
    
    # Create and fit a simple pipeline
    config = TrainConfig()
    pipeline = _build_pipeline(config)
    train_genes = splits["train_genes"]
    pipeline.fit(features.loc[train_genes], labels.loc[train_genes]["label"])
    
    # Test prediction collection
    test_genes = splits["test_genes"]
    predictions = _collect_predictions(pipeline, features, test_genes, "test")
    
    assert isinstance(predictions, pd.DataFrame)
    assert list(predictions.columns) == ["gene", "split", "y_prob"]
    assert len(predictions) == len(test_genes)
    assert (predictions["split"] == "test").all()
    assert predictions["gene"].tolist() == test_genes
    assert all(0 <= prob <= 1 for prob in predictions["y_prob"])


def test_compute_metrics():
    """Test _compute_metrics function."""
    # Create test data
    labels = pd.Series([0, 1, 0, 1, 1], index=["G1", "G2", "G3", "G4", "G5"])
    pred_frame = pd.DataFrame({
        "gene": ["G1", "G2", "G3", "G4", "G5"],
        "y_prob": [0.1, 0.9, 0.2, 0.8, 0.7]
    })
    
    metrics = _compute_metrics(labels, pred_frame)
    
    assert isinstance(metrics, dict)
    assert "auroc" in metrics
    assert "ap" in metrics
    assert 0 <= metrics["auroc"] <= 1
    assert 0 <= metrics["ap"] <= 1


def test_train_model_full_pipeline(tmp_path):
    """Test full train_model pipeline."""
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    
    # Create test data
    features, labels, splits = create_test_processed_data(processed_dir)
    
    # Train model
    config = TrainConfig(seed=42)
    result = train_model(
        processed_dir=processed_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        config=config
    )
    
    # Check result structure
    assert isinstance(result, TrainResult)
    assert isinstance(result.pipeline, Pipeline)
    assert isinstance(result.train_predictions, pd.DataFrame)
    assert isinstance(result.test_predictions, pd.DataFrame)
    assert isinstance(result.train_metrics, dict)
    assert isinstance(result.test_metrics, dict)
    assert result.dataset_hash == "test_hash_123"
    
    # Check metrics
    assert "auroc" in result.train_metrics
    assert "ap" in result.train_metrics
    assert "auroc" in result.test_metrics
    assert "ap" in result.test_metrics
    
    # Check saved files
    assert (models_dir / "logreg_pipeline.pkl").exists()
    assert (models_dir / "feature_list.json").exists()
    assert (reports_dir / "predictions.parquet").exists()
    assert (reports_dir / "metrics_basic.json").exists()
    
    # Verify saved model can be loaded
    loaded_pipeline = joblib.load(models_dir / "logreg_pipeline.pkl")
    assert isinstance(loaded_pipeline, Pipeline)
    
    # Verify feature list
    with open(models_dir / "feature_list.json") as f:
        feature_data = json.load(f)
    assert "feature_order" in feature_data
    assert feature_data["feature_order"] == list(features.columns)
    
    # Verify predictions file
    saved_preds = pd.read_parquet(reports_dir / "predictions.parquet")
    assert "gene" in saved_preds.columns
    assert "split" in saved_preds.columns
    assert "y_prob" in saved_preds.columns
    assert "y_true" in saved_preds.columns
    
    # Verify metrics file
    with open(reports_dir / "metrics_basic.json") as f:
        metrics_data = json.load(f)
    assert "train" in metrics_data
    assert "test" in metrics_data
    assert "train_size" in metrics_data
    assert "test_size" in metrics_data


def test_train_model_default_config(tmp_path):
    """Test train_model with default config."""
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    
    # Create test data
    create_test_processed_data(processed_dir)
    
    # Train with default config (None)
    result = train_model(
        processed_dir=processed_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        config=None
    )
    
    assert isinstance(result, TrainResult)
    # Should use default config values
    clf = result.pipeline.steps[1][1]
    assert clf.C == 1.0  # Default TrainConfig.C


def test_train_model_missing_processed_data(tmp_path):
    """Test train_model with missing processed data."""
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    
    processed_dir.mkdir()  # Empty directory
    
    with pytest.raises(TrainingError):
        train_model(
            processed_dir=processed_dir,
            models_dir=models_dir,
            reports_dir=reports_dir
        )


def test_train_result_dataclass():
    """Test TrainResult dataclass structure."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    # Create mock objects
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression())
    ])
    
    train_preds = pd.DataFrame({"gene": ["G1"], "split": ["train"], "y_prob": [0.5]})
    test_preds = pd.DataFrame({"gene": ["G2"], "split": ["test"], "y_prob": [0.7]})
    train_metrics = {"auroc": 0.8, "ap": 0.75}
    test_metrics = {"auroc": 0.85, "ap": 0.8}
    dataset_hash = "test_hash"
    
    result = TrainResult(
        pipeline=pipeline,
        train_predictions=train_preds,
        test_predictions=test_preds,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        dataset_hash=dataset_hash
    )
    
    assert result.pipeline == pipeline
    assert result.train_predictions.equals(train_preds)
    assert result.test_predictions.equals(test_preds)
    assert result.train_metrics == train_metrics
    assert result.test_metrics == test_metrics
    assert result.dataset_hash == dataset_hash


def test_predictions_include_true_labels(tmp_path):
    """Test that final predictions include true labels."""
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    
    features, labels, splits = create_test_processed_data(processed_dir)
    
    result = train_model(
        processed_dir=processed_dir,
        models_dir=models_dir,
        reports_dir=reports_dir
    )
    
    # Check that predictions have y_true column
    saved_preds = pd.read_parquet(reports_dir / "predictions.parquet")
    assert "y_true" in saved_preds.columns
    
    # Verify y_true values match labels
    for _, row in saved_preds.iterrows():
        gene = row["gene"]
        expected_label = labels.loc[gene, "label"]
        assert row["y_true"] == expected_label


def test_model_constants():
    """Test model module constants."""
    from oncotarget_lite.model import _DEF_MODEL_NAME, _DEF_PREDICTIONS, _DEF_FEATURES
    
    assert _DEF_MODEL_NAME == "logreg_pipeline.pkl"
    assert _DEF_PREDICTIONS == "predictions.parquet"
    assert _DEF_FEATURES == "feature_list.json"