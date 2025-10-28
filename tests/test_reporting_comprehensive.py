"""Comprehensive tests for reporting.py to boost coverage."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import tempfile
from unittest.mock import Mock, patch

from oncotarget_lite.reporting import (
    ReportingError, _load_metrics, _load_bootstrap, _load_predictions,
    _rank_test_predictions, _load_shap_arrays, _list_items, _describe_gene,
    _metric_table_html, _metric_table_markdown, _update_model_card,
    _update_readme, _mlflow_link, generate_scorecard, build_docs_index,
    SCORECARD_PATH, DOCS_INDEX, RUN_CONTEXT, README_PATH
)
from oncotarget_lite.utils import save_json, save_dataframe, write_text


def create_test_reports_data(reports_dir: Path):
    """Create test reports data."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metrics.json
    metrics = {
        "auroc": 0.85,
        "ap": 0.78,
        "brier": 0.15,
        "ece": 0.05,
        "accuracy": 0.82,
        "f1": 0.75,
        "train_auroc": 0.87,
        "test_auroc": 0.85,
        "overfit_gap": 0.02
    }
    save_json(reports_dir / "metrics.json", metrics)
    
    # Create bootstrap.json
    bootstrap = {
        "auroc": {"lower": 0.80, "upper": 0.90},
        "ap": {"lower": 0.70, "upper": 0.85}
    }
    save_json(reports_dir / "bootstrap.json", bootstrap)
    
    # Create predictions.parquet
    predictions = pd.DataFrame({
        "gene": ["GENE001", "GENE002", "GENE003", "GENE004", "GENE005"],
        "split": ["train", "train", "test", "test", "test"],
        "y_prob": [0.1, 0.9, 0.8, 0.3, 0.6],
        "y_true": [0, 1, 1, 0, 1]
    })
    save_dataframe(reports_dir / "predictions.parquet", predictions)
    
    return metrics, bootstrap, predictions


def create_test_shap_data(shap_dir: Path):
    """Create test SHAP data."""
    shap_dir.mkdir(parents=True, exist_ok=True)
    
    # Create alias_map.json
    alias_map = {
        "GENE1": "GENE003",
        "GENE2": "GENE004",
        "GENE3": "GENE005"
    }
    alias_path = shap_dir / "alias_map.json"
    alias_path.write_text(json.dumps(alias_map), encoding="utf-8")
    
    # Create shap_values.npz
    genes = ["GENE003", "GENE004", "GENE005"]
    feature_names = ["feature1", "feature2", "feature3"]
    values = np.random.randn(3, 3)  # 3 genes, 3 features
    
    np.savez(
        shap_dir / "shap_values.npz",
        genes=np.array(genes, dtype=object),
        values=values,
        feature_names=np.array(feature_names, dtype=object)
    )
    
    return alias_map, genes, feature_names, values


def test_reporting_error():
    """Test ReportingError exception."""
    error = ReportingError("Test reporting error")
    assert str(error) == "Test reporting error"
    assert isinstance(error, RuntimeError)


def test_load_metrics_missing_file(tmp_path):
    """Test _load_metrics with missing file."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    
    with pytest.raises(ReportingError) as exc_info:
        _load_metrics(reports_dir)
    
    assert "metrics.json not found" in str(exc_info.value)


def test_load_metrics_success(tmp_path):
    """Test _load_metrics with valid file."""
    reports_dir = tmp_path / "reports"
    metrics, _, _ = create_test_reports_data(reports_dir)
    
    loaded_metrics = _load_metrics(reports_dir)
    assert loaded_metrics == metrics


def test_load_bootstrap_missing_file(tmp_path):
    """Test _load_bootstrap with missing file."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    
    with pytest.raises(ReportingError) as exc_info:
        _load_bootstrap(reports_dir)
    
    assert "bootstrap.json not found" in str(exc_info.value)


def test_load_bootstrap_success(tmp_path):
    """Test _load_bootstrap with valid file."""
    reports_dir = tmp_path / "reports"
    _, bootstrap, _ = create_test_reports_data(reports_dir)
    
    loaded_bootstrap = _load_bootstrap(reports_dir)
    assert loaded_bootstrap == bootstrap


def test_load_predictions_missing_file(tmp_path):
    """Test _load_predictions with missing file."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    
    with pytest.raises(ReportingError) as exc_info:
        _load_predictions(reports_dir)
    
    assert "predictions.parquet missing" in str(exc_info.value)


def test_load_predictions_success(tmp_path):
    """Test _load_predictions with valid file."""
    reports_dir = tmp_path / "reports"
    _, _, predictions = create_test_reports_data(reports_dir)
    
    loaded_predictions = _load_predictions(reports_dir)
    pd.testing.assert_frame_equal(loaded_predictions, predictions)


def test_rank_test_predictions():
    """Test _rank_test_predictions function."""
    preds = pd.DataFrame({
        "gene": ["GENE001", "GENE002", "GENE003", "GENE004"],
        "split": ["train", "test", "test", "test"],
        "y_prob": [0.1, 0.8, 0.3, 0.6],
        "y_true": [0, 1, 0, 1]
    })
    
    ranked = _rank_test_predictions(preds)
    
    # Should only include test predictions
    assert len(ranked) == 3
    assert all(ranked.index.isin(["GENE002", "GENE003", "GENE004"]))
    
    # Should be sorted by y_prob descending
    expected_order = ["GENE002", "GENE004", "GENE003"]  # 0.8, 0.6, 0.3
    assert ranked.index.tolist() == expected_order
    
    # Check rank and percentile columns
    assert "rank" in ranked.columns
    assert "percentile" in ranked.columns
    assert ranked.loc["GENE002", "rank"] == 1
    assert ranked.loc["GENE004", "rank"] == 2
    assert ranked.loc["GENE003", "rank"] == 3


def test_rank_test_predictions_no_test_data():
    """Test _rank_test_predictions with no test data."""
    preds = pd.DataFrame({
        "gene": ["GENE001", "GENE002"],
        "split": ["train", "train"],
        "y_prob": [0.1, 0.8],
        "y_true": [0, 1]
    })
    
    with pytest.raises(ReportingError) as exc_info:
        _rank_test_predictions(preds)
    
    assert "No test predictions available" in str(exc_info.value)


def test_load_shap_arrays_missing_files(tmp_path):
    """Test _load_shap_arrays with missing files."""
    shap_dir = tmp_path / "shap"
    shap_dir.mkdir()
    
    with pytest.raises(ReportingError) as exc_info:
        _load_shap_arrays(shap_dir)
    
    assert "SHAP artefacts missing" in str(exc_info.value)


def test_load_shap_arrays_success(tmp_path):
    """Test _load_shap_arrays with valid files."""
    shap_dir = tmp_path / "shap"
    expected_alias_map, expected_genes, expected_features, expected_values = create_test_shap_data(shap_dir)
    
    alias_map, shap_lookup, feature_names = _load_shap_arrays(shap_dir)
    
    assert alias_map == expected_alias_map
    assert feature_names == expected_features
    assert set(shap_lookup.keys()) == set(expected_genes)
    
    # Check that values are correctly mapped
    for i, gene in enumerate(expected_genes):
        np.testing.assert_array_equal(shap_lookup[gene], expected_values[i])


def test_list_items():
    """Test _list_items function."""
    series = pd.Series([0.5, -0.3, 0.1], index=["feat1", "feat2", "feat3"])
    result = _list_items(series)
    
    expected = (
        "<li><strong>feat1</strong>: +0.500</li>"
        "<li><strong>feat2</strong>: -0.300</li>"
        "<li><strong>feat3</strong>: +0.100</li>"
    )
    assert result == expected


def test_describe_gene_missing_prediction():
    """Test _describe_gene with missing prediction."""
    ranked_preds = pd.DataFrame({"y_prob": [0.5], "rank": [1], "percentile": [1.0]}, index=["GENE001"])
    shap_lookup = {"GENE002": np.array([0.1, 0.2, 0.3])}
    feature_names = ["f1", "f2", "f3"]
    
    with pytest.raises(ReportingError) as exc_info:
        _describe_gene("GENE002", "ALIAS1", ranked_preds, shap_lookup, feature_names)
    
    assert "Gene GENE002 missing from predictions" in str(exc_info.value)


def test_describe_gene_missing_shap():
    """Test _describe_gene with missing SHAP data."""
    ranked_preds = pd.DataFrame({"y_prob": [0.5], "rank": [1], "percentile": [1.0]}, index=["GENE001"])
    shap_lookup = {"GENE002": np.array([0.1, 0.2, 0.3])}
    feature_names = ["f1", "f2", "f3"]
    
    with pytest.raises(ReportingError) as exc_info:
        _describe_gene("GENE001", "ALIAS1", ranked_preds, shap_lookup, feature_names)
    
    assert "Gene GENE001 missing from SHAP values" in str(exc_info.value)


def test_describe_gene_success():
    """Test _describe_gene with valid data."""
    ranked_preds = pd.DataFrame({
        "y_prob": [0.75], 
        "rank": [1], 
        "percentile": [0.95]
    }, index=["GENE001"])
    
    shap_lookup = {"GENE001": np.array([0.5, -0.3, 0.1, -0.8, 0.2])}
    feature_names = ["f1", "f2", "f3", "f4", "f5"]
    
    result = _describe_gene("GENE001", "ALIAS1", ranked_preds, shap_lookup, feature_names)
    
    assert "GENE001 (ALIAS1)" in result
    assert "Predicted score: <strong>0.750</strong>" in result
    assert "Rank: 1" in result
    assert "Percentile: 95.0%" in result
    assert "Top positive contributors" in result
    assert "Top negative contributors" in result
    assert "shap/example_ALIAS1.png" in result


def test_metric_table_html():
    """Test _metric_table_html function."""
    metrics = {
        "auroc": 0.85,
        "ap": 0.78,
        "brier": 0.15,
        "ece": 0.05,
        "accuracy": 0.82,
        "f1": 0.75,
        "train_auroc": 0.87,
        "test_auroc": 0.85,
        "overfit_gap": 0.02
    }
    bootstrap = {
        "auroc": {"lower": 0.80, "upper": 0.90},
        "ap": {"lower": 0.70, "upper": 0.85}
    }
    
    result = _metric_table_html(metrics, bootstrap)
    
    assert "<table>" in result
    assert "<tr><th>Metric</th><th>Value</th><th>95% CI</th></tr>" in result
    assert "<td>AUROC</td><td>0.850</td><td>0.800 – 0.900</td>" in result
    assert "<td>Average Precision</td><td>0.780</td><td>0.700 – 0.850</td>" in result
    assert "<td>Brier</td><td>0.150</td><td>–</td>" in result


def test_metric_table_markdown():
    """Test _metric_table_markdown function."""
    metrics = {
        "auroc": 0.85,
        "ap": 0.78,
        "brier": 0.15,
        "ece": 0.05,
        "accuracy": 0.82,
        "f1": 0.75,
        "train_auroc": 0.87,
        "test_auroc": 0.85,
        "overfit_gap": 0.02
    }
    bootstrap = {
        "auroc": {"lower": 0.80, "upper": 0.90},
        "ap": {"lower": 0.70, "upper": 0.85}
    }
    
    result = _metric_table_markdown(metrics, bootstrap)
    
    assert "| Metric | Value | 95% CI |" in result
    assert "| --- | --- | --- |" in result
    assert "| AUROC | 0.850 | 0.800 – 0.900 |" in result
    assert "| Average Precision | 0.780 | 0.700 – 0.850 |" in result
    assert "| Brier | 0.150 | – |" in result


def test_update_model_card_missing_file(tmp_path):
    """Test _update_model_card with missing file."""
    model_card = tmp_path / "model_card.md"
    table = "| Test | Table |"
    
    # Should not raise error when file doesn't exist
    _update_model_card(model_card, table)
    assert not model_card.exists()


def test_update_model_card_missing_markers(tmp_path):
    """Test _update_model_card with missing markers."""
    model_card = tmp_path / "model_card.md"
    content = "# Model Card\n\nSome content without markers."
    model_card.write_text(content, encoding="utf-8")
    
    table = "| Test | Table |"
    _update_model_card(model_card, table)
    
    # Content should remain unchanged
    assert model_card.read_text(encoding="utf-8") == content


def test_update_model_card_success(tmp_path):
    """Test _update_model_card with valid markers."""
    model_card = tmp_path / "model_card.md"
    content = """# Model Card

Some intro text.

<!-- METRICS_TABLE_START -->
Old table content
<!-- METRICS_TABLE_END -->

Some footer text."""
    
    model_card.write_text(content, encoding="utf-8")
    
    table = "| New | Table |\n| --- | --- |\n| Test | Data |"
    _update_model_card(model_card, table)
    
    updated_content = model_card.read_text(encoding="utf-8")
    assert "Old table content" not in updated_content
    assert "| New | Table |" in updated_content
    assert "Some intro text." in updated_content
    assert "Some footer text." in updated_content


def test_update_readme_missing_file(tmp_path, monkeypatch):
    """Test _update_readme with missing file."""
    # Temporarily change README_PATH
    fake_readme = tmp_path / "README.md"
    monkeypatch.setattr("oncotarget_lite.reporting.README_PATH", fake_readme)
    
    table = "| Test | Table |"
    _update_readme(table)
    
    assert not fake_readme.exists()


def test_update_readme_success(tmp_path, monkeypatch):
    """Test _update_readme with valid markers."""
    fake_readme = tmp_path / "README.md"
    monkeypatch.setattr("oncotarget_lite.reporting.README_PATH", fake_readme)
    
    content = """# Project

<!-- README_METRICS_START -->
Old metrics
<!-- README_METRICS_END -->

Footer."""
    
    fake_readme.write_text(content, encoding="utf-8")
    
    table = "| New | Metrics |"
    _update_readme(table)
    
    updated_content = fake_readme.read_text(encoding="utf-8")
    assert "Old metrics" not in updated_content
    assert "| New | Metrics |" in updated_content


def test_mlflow_link_missing_file(tmp_path, monkeypatch):
    """Test _mlflow_link with missing run context file."""
    fake_context = tmp_path / "run_context.json"
    monkeypatch.setattr("oncotarget_lite.reporting.RUN_CONTEXT", fake_context)
    
    result = _mlflow_link()
    assert result is None


def test_mlflow_link_missing_run_id(tmp_path, monkeypatch):
    """Test _mlflow_link with missing run_id."""
    fake_context = tmp_path / "run_context.json"
    monkeypatch.setattr("oncotarget_lite.reporting.RUN_CONTEXT", fake_context)
    
    context = {"experiment_name": "test"}
    fake_context.write_text(json.dumps(context), encoding="utf-8")
    
    result = _mlflow_link()
    assert result is None


def test_mlflow_link_success(tmp_path, monkeypatch):
    """Test _mlflow_link with valid run_id."""
    fake_context = tmp_path / "run_context.json"
    monkeypatch.setattr("oncotarget_lite.reporting.RUN_CONTEXT", fake_context)
    
    context = {"run_id": "test-run-123"}
    fake_context.write_text(json.dumps(context), encoding="utf-8")
    
    result = _mlflow_link()
    assert result == "mlflow://runs/test-run-123"


def test_generate_scorecard_success(tmp_path):
    """Test generate_scorecard with valid data."""
    reports_dir = tmp_path / "reports"
    shap_dir = tmp_path / "reports" / "shap"
    output_path = tmp_path / "scorecard.html"
    
    # Create test data
    create_test_reports_data(reports_dir)
    create_test_shap_data(shap_dir)
    
    result_path = generate_scorecard(
        reports_dir=reports_dir,
        shap_dir=shap_dir,
        output_path=output_path
    )
    
    assert result_path == output_path
    assert output_path.exists()
    
    content = output_path.read_text(encoding="utf-8")
    assert "<html>" in content
    assert "oncotarget-lite Scorecard" in content
    assert "AUROC" in content
    assert "GENE003 (GENE1)" in content  # Should include gene descriptions


def test_build_docs_index_success(tmp_path):
    """Test build_docs_index with valid data."""
    reports_dir = tmp_path / "reports"
    docs_dir = tmp_path / "docs"
    model_card = tmp_path / "model_card.md"
    
    # Create test data
    create_test_reports_data(reports_dir)
    
    # Create model card with markers
    model_card_content = """# Model Card

<!-- METRICS_TABLE_START -->
Old content
<!-- METRICS_TABLE_END -->

End."""
    model_card.write_text(model_card_content, encoding="utf-8")
    
    result_path = build_docs_index(
        reports_dir=reports_dir,
        docs_dir=docs_dir,
        model_card=model_card
    )
    
    expected_path = docs_dir / "index.html"
    assert result_path == expected_path
    assert expected_path.exists()
    
    content = expected_path.read_text(encoding="utf-8")
    assert "oncotarget-lite Overview" in content
    assert "AUROC" in content
    assert "Target scorecard" in content
    assert "Calibration" in content
    
    # Check that model card was updated
    updated_model_card = model_card.read_text(encoding="utf-8")
    assert "Old content" not in updated_model_card
    assert "| AUROC |" in updated_model_card


def test_reporting_constants():
    """Test reporting module constants."""
    assert str(SCORECARD_PATH) == "reports/target_scorecard.html"
    assert str(DOCS_INDEX) == "docs/index.html"
    assert str(RUN_CONTEXT) == "reports/run_context.json"
    assert str(README_PATH) == "README.md"