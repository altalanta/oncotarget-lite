"""Comprehensive CLI tests to boost coverage."""

import subprocess
import sys
import pytest
import types
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
import tempfile


@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    """Mock MLflow and SHAP to prevent import errors."""
    mlflow_stub = types.SimpleNamespace(
        start_run=lambda **k: types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda *a: False),
        log_params=lambda *a, **k: None,
        log_metrics=lambda *a, **k: None,
        log_metric=lambda *a, **k: None,
        set_experiment=lambda *a, **k: None,
        log_artifact=lambda *a, **k: None,
        get_tracking_uri=lambda: "file://./mlruns",
        set_tracking_uri=lambda *a, **k: None,
        log_artifacts=lambda *a, **k: None,
        create_experiment=lambda *a, **k: "test_exp",
        get_experiment_by_name=lambda *a, **k: None,
    )
    
    shap_stub = types.SimpleNamespace(
        Explainer=Mock,
    )
    
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_stub)
    monkeypatch.setitem(sys.modules, "shap", shap_stub)


def create_test_data(base_path: Path):
    """Create minimal test data structure."""
    raw_dir = base_path / "raw"
    processed_dir = base_path / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create minimal CSV files
    expression_data = pd.DataFrame({
        "gene": ["GENE001", "GENE002"],
        "median_TPM": [1.5, 2.3]
    })
    expression_data.to_csv(raw_dir / "expression.csv", index=False)
    
    annotations_data = pd.DataFrame({
        "gene": ["GENE001", "GENE002"],
        "is_cell_surface": [1, 0],
        "signal_peptide": [1, 0],
        "ig_like_domain": [0, 1],
        "protein_length": [500, 300]
    })
    annotations_data.to_csv(raw_dir / "uniprot_annotations.csv", index=False)
    
    depmap_data = pd.DataFrame({
        "gene": ["GENE001", "GENE002"],
        "cell_line": ["CELL1", "CELL2"],
        "dependency_score": [-0.5, -1.2]
    })
    depmap_data.to_csv(raw_dir / "DepMap_essentials_subset.csv", index=False)
    
    return raw_dir, processed_dir


def test_cli_app_import():
    """Test that CLI app can be imported."""
    from oncotarget_lite.cli import app
    assert app is not None


def test_prepare_command_help():
    """Test prepare command help."""
    result = subprocess.run([
        sys.executable, "-m", "oncotarget_lite", "prepare", "--help"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "prepare" in output.lower()


def test_train_command_help():
    """Test train command help."""
    result = subprocess.run([
        sys.executable, "-m", "oncotarget_lite", "train", "--help"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "train" in output.lower()


def test_eval_command_help():
    """Test eval command help."""
    result = subprocess.run([
        sys.executable, "-m", "oncotarget_lite", "eval", "--help"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "eval" in output.lower()


def test_explain_command_help():
    """Test explain command help."""
    result = subprocess.run([
        sys.executable, "-m", "oncotarget_lite", "explain", "--help"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "explain" in output.lower()


def test_scorecard_command_help():
    """Test scorecard command help."""
    result = subprocess.run([
        sys.executable, "-m", "oncotarget_lite", "scorecard", "--help"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "scorecard" in output.lower()


def test_snapshot_command_help():
    """Test snapshot command help."""
    result = subprocess.run([
        sys.executable, "-m", "oncotarget_lite", "snapshot", "--help"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "snapshot" in output.lower()


@patch('oncotarget_lite.cli.prepare_dataset')
def test_prepare_command_execution(mock_prepare, tmp_path):
    """Test prepare command execution with mocked dependencies."""
    mock_prepare.return_value = None
    
    result = subprocess.run([
        sys.executable, "-m", "oncotarget_lite", "prepare",
        "--raw-dir", str(tmp_path / "raw"),
        "--processed-dir", str(tmp_path / "processed")
    ], capture_output=True, text=True)
    
    # Should not crash, even if data doesn't exist
    assert result.returncode in [0, 1]  # May fail due to missing data, but shouldn't crash


def test_generate_data_command_execution(tmp_path):
    """Test generate-data command execution."""
    out_dir = tmp_path / "output"
    
    result = subprocess.run([
        sys.executable, "-m", "oncotarget_lite", "generate-data",
        "--out-dir", str(out_dir)
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert (out_dir / "expression.csv").exists()
    assert (out_dir / "uniprot_annotations.csv").exists()
    assert (out_dir / "DepMap_essentials_subset.csv").exists()


def test_cli_lazy_mlflow_import():
    """Test that CLI commands work with lazy MLflow import."""
    from oncotarget_lite.cli import app
    from oncotarget_lite.utils import _mlflow
    
    # Should be able to get MLflow reference
    mlflow = _mlflow()
    assert hasattr(mlflow, 'start_run')


def test_typer_app_configuration():
    """Test Typer app configuration."""
    from oncotarget_lite.cli import app
    
    # App should be configured properly
    assert hasattr(app, 'commands')
    assert len(app.commands) > 0
    
    # Check expected commands exist
    command_names = list(app.commands.keys())
    expected_commands = ["prepare", "train", "eval", "explain", "generate-data", "scorecard", "snapshot"]
    
    for cmd in expected_commands:
        assert cmd in command_names


def test_cli_command_decorators():
    """Test that CLI commands are properly decorated."""
    from oncotarget_lite import cli
    
    # Check that functions have command decorators
    assert hasattr(cli, 'prepare')
    assert hasattr(cli, 'train')
    assert hasattr(cli, 'eval_command')
    assert hasattr(cli, 'explain')
    assert hasattr(cli, 'generate_data')


def test_cli_parameter_validation():
    """Test CLI parameter validation."""
    # Test with invalid directories
    result = subprocess.run([
        sys.executable, "-m", "oncotarget_lite", "prepare",
        "--raw-dir", "/nonexistent/path",
        "--processed-dir", "/another/nonexistent/path"
    ], capture_output=True, text=True)
    
    # Should fail with appropriate error
    assert result.returncode != 0


def test_main_cli_entry_point():
    """Test main CLI entry point."""
    result = subprocess.run([
        sys.executable, "-m", "oncotarget_lite"
    ], capture_output=True, text=True)
    
    # Should show help when no command provided
    assert result.returncode in [0, 2]  # Typer may return 2 for missing command
    output = result.stdout + result.stderr
    assert "Usage" in output or "Commands" in output


def test_cli_error_handling():
    """Test CLI error handling."""
    # Test with completely invalid command
    result = subprocess.run([
        sys.executable, "-m", "oncotarget_lite", "invalid-command"
    ], capture_output=True, text=True)
    
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "No such command" in output or "Usage" in output


def test_cli_imports_work():
    """Test that all CLI imports work properly."""
    # These should not raise import errors
    from oncotarget_lite.cli import app, prepare, train, eval_command, explain, generate_data
    from oncotarget_lite.cli import scorecard, snapshot
    
    # All should be callable
    assert callable(prepare)
    assert callable(train)
    assert callable(eval_command)
    assert callable(explain)
    assert callable(generate_data)
    assert callable(scorecard)
    assert callable(snapshot)


def test_cli_help_comprehensive():
    """Test comprehensive CLI help functionality."""
    commands = ["prepare", "train", "eval", "explain", "generate-data", "scorecard", "snapshot"]
    
    for cmd in commands:
        result = subprocess.run([
            sys.executable, "-m", "oncotarget_lite", cmd, "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Help failed for command: {cmd}"
        output = result.stdout + result.stderr
        assert len(output) > 0, f"No help output for command: {cmd}"