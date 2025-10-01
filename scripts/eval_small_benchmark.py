"""Deterministic, tiny benchmark to keep CI fast."""

from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from oncotarget_lite.evaluation.bootstrap import bootstrap_ci, compute_classification_metrics


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for deterministic behavior."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Set torch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def generate_synthetic_dataset(n_samples: int = 1500, seed: int = 1337) -> tuple[np.ndarray, np.ndarray]:
    """Generate a tiny synthetic binary classification dataset."""
    set_all_seeds(seed)
    
    # Create a dataset that mimics the oncology task characteristics
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=1,
        class_sep=0.8,
        flip_y=0.05,  # Add some label noise
        random_state=seed,
    )
    
    return X, y


def run_benchmark(k_folds: int = 5, base_seed: int = 1337) -> dict:
    """Run deterministic benchmark with K-fold cross-validation."""
    # Generate synthetic dataset
    X, y = generate_synthetic_dataset(seed=base_seed)
    
    # Store per-seed metrics
    per_seed_metrics = []
    
    # Use different seeds for each fold to estimate variance
    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=base_seed)
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        fold_seed = base_seed + fold_idx
        set_all_seeds(fold_seed)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train minimal logistic regression model
        model = LogisticRegression(random_state=fold_seed, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Get predictions
        y_score = model.predict_proba(X_test)[:, 1]
        
        # Compute metrics for this fold
        metrics = compute_classification_metrics(y_test, y_score)
        metrics["seed"] = fold_seed
        metrics["fold"] = fold_idx
        
        per_seed_metrics.append(metrics)
    
    return {
        "dataset": "tiny-synthetic",
        "n_samples": len(X),
        "n_folds": k_folds,
        "base_seed": base_seed,
        "git_sha": get_git_sha(),
        "per_seed_metrics": per_seed_metrics,
    }


def compute_summary_metrics(results: dict) -> dict:
    """Compute aggregated metrics with 95% confidence intervals."""
    per_seed_metrics = results["per_seed_metrics"]
    
    # Extract metric values across seeds
    metric_names = ["auroc", "auprc", "accuracy", "ece"]
    summary_metrics = {}
    
    for metric_name in metric_names:
        values = np.array([m[metric_name] for m in per_seed_metrics])
        mean_val = np.mean(values)
        ci_low, ci_high = bootstrap_ci(values, alpha=0.05, n_boot=2000, seed=1337)
        
        summary_metrics[metric_name] = {
            "mean": float(mean_val),
            "ci95": [float(ci_low), float(ci_high)],
        }
    
    return {
        "dataset": results["dataset"],
        "n": results["n_folds"],
        "git_sha": results["git_sha"],
        "metrics": summary_metrics,
    }


def main() -> None:
    """Run the benchmark and save results."""
    print("Running deterministic tiny benchmark...")
    
    # Run benchmark
    results = run_benchmark()
    
    # Compute summary with confidence intervals
    summary = compute_summary_metrics(results)
    
    # Ensure docs directory exists
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Save raw results
    raw_path = docs_dir / "metrics_raw.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Raw metrics saved to {raw_path}")
    
    # Save summary results
    summary_path = docs_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary metrics saved to {summary_path}")
    
    # Print summary
    print("\nBenchmark Results:")
    for metric_name, metric_data in summary["metrics"].items():
        mean_val = metric_data["mean"]
        ci_low, ci_high = metric_data["ci95"]
        print(f"  {metric_name.upper()}: {mean_val:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}])")


if __name__ == "__main__":
    main()