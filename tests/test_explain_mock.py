"""Tests for explain.py with mocked SHAP to avoid deprecation warnings."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sys
import types

# Mock SHAP before importing explain module
mock_shap = types.SimpleNamespace(
    Explainer=Mock,
)

with patch.dict('sys.modules', {'shap': mock_shap}):
    from oncotarget_lite.explain import (
        ShapArtifacts, ExplanationError, _load_training_state,
        _select_examples, _plot_global, _plot_gene, generate_shap,
        EXAMPLE_ALIASES
    )

from oncotarget_lite.utils import save_json
import joblib


def create_mock_trained_model(processed_dir: Path, models_dir: Path):
    """Create mock trained model and data for testing."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create features
    features = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "feature2": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        "feature3": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    }, index=["GENE001", "GENE002", "GENE003", "GENE004", "GENE005", "GENE006"])
    
    # Create splits
    splits = {
        "train_genes": ["GENE001", "GENE002", "GENE003"],
        "test_genes": ["GENE004", "GENE005", "GENE006"],
        "dataset_hash": "test_hash_123"
    }
    
    # Create and save a simple model
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42))
    ])
    
    # Fit with dummy data
    X_train = features.loc[splits["train_genes"]]
    y_train = pd.Series([0, 1, 0], index=splits["train_genes"])
    pipeline.fit(X_train, y_train)
    
    # Save files
    features.to_parquet(processed_dir / "features.parquet")
    save_json(processed_dir / "splits.json", splits)
    joblib.dump(pipeline, models_dir / "logreg_pipeline.pkl")
    
    return features, splits, pipeline


def test_shap_artifacts_dataclass():
    """Test ShapArtifacts dataclass structure."""
    global_importance = pd.Series([0.5, 0.3, 0.2], index=["f1", "f2", "f3"])
    per_gene_contribs = {
        "GENE1": pd.Series([0.1, 0.2, 0.3], index=["f1", "f2", "f3"])
    }
    alias_map = {"ALIAS1": "GENE1"}
    
    artifacts = ShapArtifacts(
        global_importance=global_importance,
        per_gene_contribs=per_gene_contribs,
        alias_map=alias_map
    )
    
    assert artifacts.global_importance.equals(global_importance)
    assert artifacts.per_gene_contribs == per_gene_contribs
    assert artifacts.alias_map == alias_map


def test_explanation_error():
    """Test ExplanationError exception."""
    error = ExplanationError("Test explanation error")
    assert str(error) == "Test explanation error"
    assert isinstance(error, RuntimeError)


def test_load_training_state_missing_model(tmp_path):
    """Test _load_training_state with missing model."""
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    processed_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    
    # Create features and splits but no model
    features = pd.DataFrame({"f1": [1, 2]}, index=["G1", "G2"])
    splits = {"train_genes": ["G1"], "test_genes": ["G2"]}
    
    features.to_parquet(processed_dir / "features.parquet")
    save_json(processed_dir / "splits.json", splits)
    
    with pytest.raises(ExplanationError) as exc_info:
        _load_training_state(processed_dir, models_dir)
    
    assert "Trained model not found" in str(exc_info.value)


def test_load_training_state_success(tmp_path):
    """Test _load_training_state with valid files."""
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    
    expected_features, expected_splits, expected_pipeline = create_mock_trained_model(
        processed_dir, models_dir
    )
    
    features, splits, pipeline = _load_training_state(processed_dir, models_dir)
    
    assert features.equals(expected_features)
    assert splits == expected_splits
    assert isinstance(pipeline, Pipeline)


def test_select_examples():
    """Test _select_examples function."""
    test_genes = ["GENE001", "GENE002", "GENE003", "GENE004", "GENE005"]
    
    # Test default k=3
    result = _select_examples(test_genes)
    assert result == ["GENE001", "GENE002", "GENE003"]
    
    # Test custom k
    result = _select_examples(test_genes, k=2)
    assert result == ["GENE001", "GENE002"]
    
    # Test k larger than list
    result = _select_examples(test_genes, k=10)
    assert result == test_genes


@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.close')
@patch('matplotlib.pyplot.subplots')
def test_plot_global(mock_subplots, mock_close, mock_savefig, tmp_path):
    """Test _plot_global function."""
    # Mock matplotlib components
    mock_fig = Mock()
    mock_ax = Mock()
    mock_subplots.return_value = (mock_fig, mock_ax)
    
    feature_names = ["feature1", "feature2", "feature3"]
    importances = np.array([0.3, 0.1, 0.5])
    path = tmp_path / "plots" / "global.png"
    
    _plot_global(feature_names, importances, path)
    
    # Verify directory was created
    assert path.parent.exists()
    
    # Verify matplotlib calls
    mock_subplots.assert_called_once_with(figsize=(6, 4))
    mock_ax.barh.assert_called_once()
    mock_ax.set_xlabel.assert_called_with("Mean |SHAP value|")
    mock_ax.set_title.assert_called_with("Global Feature Importance")
    mock_fig.tight_layout.assert_called_once()
    mock_fig.savefig.assert_called_with(path, dpi=150)
    mock_close.assert_called_once_with(mock_fig)


@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.close')
@patch('matplotlib.pyplot.subplots')
def test_plot_gene(mock_subplots, mock_close, mock_savefig, tmp_path):
    """Test _plot_gene function."""
    # Mock matplotlib components
    mock_fig = Mock()
    mock_ax = Mock()
    mock_subplots.return_value = (mock_fig, mock_ax)
    
    gene = "GENE001"
    alias = "GENE1"
    contributions = pd.Series([0.1, -0.3, 0.5, 0.2], index=["f1", "f2", "f3", "f4"])
    path = tmp_path / "plots" / "gene.png"
    
    _plot_gene(gene, alias, contributions, path)
    
    # Verify directory was created
    assert path.parent.exists()
    
    # Verify matplotlib calls
    mock_subplots.assert_called_once_with(figsize=(6, 4))
    mock_ax.barh.assert_called_once()
    mock_ax.invert_yaxis.assert_called_once()
    mock_ax.set_xlabel.assert_called_with("SHAP contribution")
    mock_ax.set_title.assert_called_with(f"{gene} ({alias}) feature contributions")
    mock_fig.tight_layout.assert_called_once()
    mock_fig.savefig.assert_called_with(path, dpi=150)
    mock_close.assert_called_once_with(mock_fig)


def test_example_aliases_constant():
    """Test EXAMPLE_ALIASES constant."""
    assert EXAMPLE_ALIASES == ["GENE1", "GENE2", "GENE3"]
    assert len(EXAMPLE_ALIASES) == 3


@patch.dict('sys.modules', {'shap': mock_shap})
@patch('oncotarget_lite.explain._plot_global')
@patch('oncotarget_lite.explain._plot_gene')
def test_generate_shap_missing_splits(mock_plot_gene, mock_plot_global, tmp_path):
    """Test generate_shap with missing train/test splits."""
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    shap_dir = tmp_path / "shap"
    
    # Create model but with empty splits
    features, splits, pipeline = create_mock_trained_model(processed_dir, models_dir)
    
    # Modify splits to be empty
    empty_splits = {"train_genes": [], "test_genes": []}
    save_json(processed_dir / "splits.json", empty_splits)
    
    with pytest.raises(ExplanationError) as exc_info:
        generate_shap(
            processed_dir=processed_dir,
            models_dir=models_dir,
            shap_dir=shap_dir
        )
    
    assert "Missing train/test splits for SHAP" in str(exc_info.value)


@patch.dict('sys.modules', {'shap': mock_shap})
@patch('oncotarget_lite.explain._plot_global')
@patch('oncotarget_lite.explain._plot_gene')
def test_generate_shap_success(mock_plot_gene, mock_plot_global, tmp_path):
    """Test successful generate_shap execution."""
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    shap_dir = tmp_path / "shap"
    
    # Create mock trained model
    features, splits, pipeline = create_mock_trained_model(processed_dir, models_dir)
    
    # Mock SHAP components
    mock_explainer = Mock()
    mock_shap_values = Mock()
    
    # Create realistic SHAP values (n_samples, n_features)
    n_test_genes = len(splits["test_genes"])
    n_features = len(features.columns)
    mock_values = np.random.randn(n_test_genes, n_features)
    
    mock_shap_values.values = mock_values
    mock_explainer.return_value = mock_shap_values
    mock_shap.Explainer.return_value = mock_explainer
    
    # Run generate_shap
    result = generate_shap(
        processed_dir=processed_dir,
        models_dir=models_dir,
        shap_dir=shap_dir,
        seed=42,
        background_size=2  # Small for test
    )
    
    # Verify result structure
    assert isinstance(result, ShapArtifacts)
    assert isinstance(result.global_importance, pd.Series)
    assert isinstance(result.per_gene_contribs, dict)
    assert isinstance(result.alias_map, dict)
    
    # Verify SHAP was called correctly
    mock_shap.Explainer.assert_called_once()
    mock_explainer.assert_called_once()
    
    # Verify plots were generated
    mock_plot_global.assert_called_once()
    assert mock_plot_gene.call_count >= 1  # Should be called for each example gene
    
    # Verify files were saved
    assert (shap_dir / "shap_values.npz").exists()
    assert (shap_dir / "alias_map.json").exists()
    
    # Verify alias map file content
    with open(shap_dir / "alias_map.json") as f:
        saved_aliases = json.load(f)
    assert isinstance(saved_aliases, dict)
    assert saved_aliases == result.alias_map


def test_reporting_constants():
    """Test explain module constants."""
    from oncotarget_lite.explain import SHAP_DIR, EXAMPLE_ALIASES
    
    assert str(SHAP_DIR) == "reports/shap"
    assert EXAMPLE_ALIASES == ["GENE1", "GENE2", "GENE3"]