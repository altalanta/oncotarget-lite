"""Tests for metrics documentation generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_eval_small_benchmark():
    """Test that eval_small_benchmark.py runs and creates expected JSON files."""
    # Run the benchmark script
    result = subprocess.run(
        [sys.executable, "-m", "scripts.eval_small_benchmark"],
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )
    
    assert result.returncode == 0, f"Benchmark script failed: {result.stderr}"
    
    # Check that JSON files were created
    docs_dir = Path("docs")
    raw_path = docs_dir / "metrics_raw.json"
    summary_path = docs_dir / "metrics_summary.json"
    
    assert raw_path.exists(), "metrics_raw.json was not created"
    assert summary_path.exists(), "metrics_summary.json was not created"
    
    # Load and validate raw results
    with open(raw_path) as f:
        raw_data = json.load(f)
    
    assert "dataset" in raw_data
    assert "per_seed_metrics" in raw_data
    assert len(raw_data["per_seed_metrics"]) == 5  # 5 folds
    
    # Validate each seed's metrics
    for seed_metrics in raw_data["per_seed_metrics"]:
        assert "auroc" in seed_metrics
        assert "auprc" in seed_metrics
        assert "accuracy" in seed_metrics
        assert "ece" in seed_metrics
        assert "seed" in seed_metrics
        assert "fold" in seed_metrics
    
    # Load and validate summary results
    with open(summary_path) as f:
        summary_data = json.load(f)
    
    assert "dataset" in summary_data
    assert "n" in summary_data
    assert "git_sha" in summary_data
    assert "metrics" in summary_data
    
    # Validate metrics have correct structure and sane values
    metrics = summary_data["metrics"]
    required_metrics = ["auroc", "auprc", "accuracy", "ece"]
    
    for metric_name in required_metrics:
        assert metric_name in metrics, f"Missing metric: {metric_name}"
        metric_data = metrics[metric_name]
        
        assert "mean" in metric_data
        assert "ci95" in metric_data
        assert len(metric_data["ci95"]) == 2
        
        mean_val = metric_data["mean"]
        ci_low, ci_high = metric_data["ci95"]
        
        # Sanity checks for metric ranges
        if metric_name in ["auroc", "auprc", "accuracy"]:
            assert 0 <= mean_val <= 1, f"{metric_name} mean out of range: {mean_val}"
            assert 0 <= ci_low <= 1, f"{metric_name} CI lower bound out of range: {ci_low}"
            assert 0 <= ci_high <= 1, f"{metric_name} CI upper bound out of range: {ci_high}"
        else:  # ECE
            assert 0 <= mean_val, f"{metric_name} mean negative: {mean_val}"
            assert 0 <= ci_low, f"{metric_name} CI lower bound negative: {ci_low}"
            assert 0 <= ci_high, f"{metric_name} CI upper bound negative: {ci_high}"
        
        assert ci_low <= ci_high, f"{metric_name} CI bounds reversed: {ci_low} > {ci_high}"


def test_render_docs_metrics():
    """Test that render_docs_metrics.py runs and updates docs/index.md correctly."""
    # Ensure benchmark results exist first
    docs_dir = Path("docs")
    summary_path = docs_dir / "metrics_summary.json"
    
    if not summary_path.exists():
        # Run benchmark first
        result = subprocess.run(
            [sys.executable, "-m", "scripts.eval_small_benchmark"],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )
        assert result.returncode == 0, f"Benchmark script failed: {result.stderr}"
    
    # Run the docs rendering script
    result = subprocess.run(
        [sys.executable, "-m", "scripts.render_docs_metrics"],
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )
    
    assert result.returncode == 0, f"Docs rendering script failed: {result.stderr}"
    
    # Check that docs/index.md was created/updated
    index_path = docs_dir / "index.md"
    assert index_path.exists(), "docs/index.md was not created"
    
    # Check that the file contains the expected markers and content
    with open(index_path) as f:
        content = f.read()
    
    assert "<!-- BEGIN: METRICS TABLE -->" in content, "Missing BEGIN marker"
    assert "<!-- END: METRICS TABLE -->" in content, "Missing END marker"
    assert "## Benchmark (deterministic CI)" in content, "Missing benchmark section header"
    assert "| Metric | Mean | 95% CI |" in content, "Missing table header"
    assert "AUROC" in content, "Missing AUROC row"
    assert "AUPRC" in content, "Missing AUPRC row"
    assert "Accuracy" in content, "Missing Accuracy row"
    assert "ECE" in content, "Missing ECE row"


def test_render_docs_metrics_without_summary_fails():
    """Test that render_docs_metrics.py fails gracefully when summary file is missing."""
    # Temporarily move summary file if it exists
    docs_dir = Path("docs")
    summary_path = docs_dir / "metrics_summary.json"
    backup_path = docs_dir / "metrics_summary.json.backup"
    
    moved_file = False
    if summary_path.exists():
        summary_path.rename(backup_path)
        moved_file = True
    
    try:
        # Run the docs rendering script (should fail)
        result = subprocess.run(
            [sys.executable, "-m", "scripts.render_docs_metrics"],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )
        
        assert result.returncode != 0, "Docs rendering should fail without summary file"
        assert "not found" in result.stdout or "not found" in result.stderr
        
    finally:
        # Restore the file if we moved it
        if moved_file and backup_path.exists():
            backup_path.rename(summary_path)