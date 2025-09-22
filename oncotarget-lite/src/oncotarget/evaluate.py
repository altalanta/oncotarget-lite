"""CLI for evaluating trained models with comprehensive metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .io import merge_gene_feature_table
from .model import MLPClassifier, MLPConfig
from .models_fttransformer import create_fttransformer_for_task
from .eval import classification_summary, summarize_bootstrap, compute_reliability_curve


def load_model_and_data(
    model_path: str,
    test_csv: Optional[str] = None,
    target_col: str = "target"
) -> tuple[nn.Module, torch.Tensor, torch.Tensor, dict]:
    """Load model and test data."""
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    model_type = checkpoint["model_type"]
    task = checkpoint["task"]
    n_features = checkpoint["n_features"]
    config = checkpoint["config"]
    
    # Create model
    if model_type == "mlp":
        # Reconstruct MLPConfig from saved config
        mlp_config = MLPConfig(
            hidden_sizes=tuple(config.get("hidden_sizes", [32, 16])),
            dropout=config.get("mlp_dropout", 0.15)
        )
        model = MLPClassifier(input_dim=n_features, config=mlp_config)
    else:  # fttransformer
        model = create_fttransformer_for_task(
            n_features=n_features,
            task=task,
            d_model=config.get("d_model", 128),
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 3),
            dropout=config.get("dropout", 0.1)
        )
    
    # Load model weights
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    # Load test data
    if test_csv:
        test_df = pd.read_csv(test_csv)
        if target_col not in test_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in {test_csv}")
        
        X_test = test_df.drop(columns=[target_col]).values
        y_test = test_df[target_col].values
    else:
        # Use built-in data
        print("Using built-in synthetic data for evaluation...")
        features_df = merge_gene_feature_table()
        
        # Create synthetic labels (same as training)
        if 'is_cell_surface' in features_df.columns:
            y_test = features_df['is_cell_surface'].astype(float).values
            features_df = features_df.drop(columns=['is_cell_surface'])
        else:
            np.random.seed(42)
            y_test = np.random.binomial(1, 0.3, len(features_df))
        
        # Select continuous features
        continuous_cols = []
        for col in features_df.columns:
            if features_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                continuous_cols.append(col)
        
        X_test = features_df[continuous_cols].values
    
    # Convert to tensors
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))
    
    model_info = {
        "model_type": model_type,
        "task": task,
        "n_features": n_features,
        "config": config
    }
    
    return model, X_test, y_test, model_info


def predict_with_model(
    model: nn.Module,
    X: torch.Tensor,
    task: str = "binary",
    batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions with the model.
    
    Returns:
        predictions: raw model outputs
        probabilities: probabilities (for binary) or same as predictions (for regression)
    """
    model.eval()
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, in loader:
            outputs = model(batch_x)
            
            all_preds.append(outputs.cpu().numpy())
            
            if task == "binary":
                probs = torch.sigmoid(outputs)
                all_probs.append(probs.cpu().numpy())
            else:  # regression
                all_probs.append(outputs.cpu().numpy())
    
    predictions = np.concatenate(all_preds)
    probabilities = np.concatenate(all_probs)
    
    return predictions, probabilities


def evaluate_binary_classification(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    bootstrap: bool = True,
    n_bootstraps: int = 256
) -> dict:
    """Comprehensive binary classification evaluation."""
    
    # Basic metrics
    metrics = classification_summary(y_true, y_scores)
    
    # Bootstrap confidence intervals
    if bootstrap:
        bootstrap_metrics = summarize_bootstrap(y_true, y_scores, n_bootstraps)
        for name, summary in bootstrap_metrics.items():
            metrics[f"{name}_bootstrap"] = {
                "mean": summary.value,
                "lower": summary.lower,
                "upper": summary.upper
            }
    
    # Calibration curve
    calibration_df = compute_reliability_curve(y_true, y_scores)
    metrics["calibration"] = calibration_df.to_dict("records")
    
    return metrics


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """Regression evaluation metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred))
    }
    
    return metrics


def compute_feature_importance(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    task: str = "binary",
    max_samples: int = 100
) -> Optional[pd.DataFrame]:
    """Compute SHAP feature importance if available."""
    try:
        from .eval import compute_shap_values
        
        # Convert back to DataFrame format for SHAP
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X.numpy(), columns=feature_names)
        
        shap_df = compute_shap_values(model, df, max_samples)
        return shap_df
    except (ImportError, Exception) as e:
        print(f"Feature importance computation failed: {e}")
        return None


def main() -> None:
    """CLI for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    
    # Required arguments
    parser.add_argument("model_path", type=str, help="Path to saved model")
    
    # Data arguments
    parser.add_argument("--test-csv", type=str, help="Test CSV file")
    parser.add_argument("--target-col", type=str, default="target", help="Target column name")
    
    # Evaluation arguments
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--bootstrap", action="store_true", help="Compute bootstrap confidence intervals")
    parser.add_argument("--n-bootstraps", type=int, default=256, help="Number of bootstrap samples")
    parser.add_argument("--feature-importance", action="store_true", help="Compute feature importance")
    parser.add_argument("--max-shap-samples", type=int, default=100, help="Max samples for SHAP computation")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--save-predictions", action="store_true", help="Save predictions to CSV")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    print("Loading model and data...")
    model, X_test, y_test, model_info = load_model_and_data(
        args.model_path, args.test_csv, args.target_col
    )
    
    print(f"Model: {model_info['model_type']} ({model_info['task']})")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {model_info['n_features']}")
    
    # Generate predictions
    print("Generating predictions...")
    predictions, probabilities = predict_with_model(
        model, X_test, model_info['task'], args.batch_size
    )
    
    # Evaluate based on task type
    if model_info['task'] == "binary":
        print("Evaluating binary classification...")
        metrics = evaluate_binary_classification(
            y_test.numpy(), 
            probabilities.squeeze(),
            bootstrap=args.bootstrap,
            n_bootstraps=args.n_bootstraps
        )
        
        print("Classification metrics:")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  AUPRC: {metrics['auprc']:.4f}")
        print(f"  Brier Score: {metrics['brier']:.4f}")
        
        if args.bootstrap:
            print("Bootstrap confidence intervals:")
            for metric_name in ['auroc', 'auprc']:
                if f"{metric_name}_bootstrap" in metrics:
                    bs_metric = metrics[f"{metric_name}_bootstrap"]
                    print(f"  {metric_name.upper()}: {bs_metric['mean']:.4f} [{bs_metric['lower']:.4f}, {bs_metric['upper']:.4f}]")
    
    else:  # regression
        print("Evaluating regression...")
        metrics = evaluate_regression(y_test.numpy(), predictions.squeeze())
        
        print("Regression metrics:")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
    
    # Feature importance
    if args.feature_importance:
        print("Computing feature importance...")
        importance_df = compute_feature_importance(
            model, X_test, y_test, model_info['task'], args.max_shap_samples
        )
        
        if importance_df is not None:
            # Save feature importance
            importance_path = output_dir / "feature_importance.csv"
            importance_df.to_csv(importance_path)
            print(f"Feature importance saved to {importance_path}")
            
            # Show top features
            mean_importance = importance_df.abs().mean().sort_values(ascending=False)
            print("Top 10 most important features:")
            for i, (feature, importance) in enumerate(mean_importance.head(10).items()):
                print(f"  {i+1}. {feature}: {importance:.4f}")
    
    # Save predictions
    if args.save_predictions:
        pred_df = pd.DataFrame({
            "true": y_test.numpy(),
            "predicted": predictions.squeeze()
        })
        
        if model_info['task'] == "binary":
            pred_df["probability"] = probabilities.squeeze()
        
        pred_path = output_dir / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"Predictions saved to {pred_path}")
    
    # Save metrics
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {metrics_path}")
    
    # Save model info
    info_path = output_dir / "model_info.json"
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"Model info saved to {info_path}")
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()