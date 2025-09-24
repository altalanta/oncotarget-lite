"""SHAP explanations for model interpretability."""

from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from .utils import set_random_seed


def get_shap_explainer(model: Any, X_background: pd.DataFrame, model_type: str = "sklearn") -> shap.Explainer:
    """Get appropriate SHAP explainer for the model type."""
    
    if model_type == "sklearn":
        # Use TreeExplainer for tree-based models, KernelExplainer otherwise
        try:
            return shap.TreeExplainer(model)
        except Exception:
            # Fallback to KernelExplainer with limited background samples
            background_sample = shap.sample(X_background, min(100, len(X_background)))
            return shap.KernelExplainer(model.predict_proba, background_sample)
    
    elif model_type == "mlp":
        # For PyTorch models, use KernelExplainer
        def model_predict(X):
            model.eval()
            with torch.no_grad():
                if isinstance(X, np.ndarray):
                    X_tensor = torch.FloatTensor(X)
                else:
                    X_tensor = torch.FloatTensor(X.values)
                outputs = torch.sigmoid(model(X_tensor)).numpy()
                # Return probabilities for both classes
                return np.column_stack([1 - outputs.flatten(), outputs.flatten()])
        
        background_sample = shap.sample(X_background, min(100, len(X_background)))
        return shap.KernelExplainer(model_predict, background_sample)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def compute_shap_values(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_type: str = "sklearn",
    random_state: int = 42
) -> tuple[shap.Explainer, np.ndarray]:
    """Compute SHAP values for test set."""
    
    set_random_seed(random_state)
    
    # Get explainer
    explainer = get_shap_explainer(model, X_train, model_type)
    
    # Compute SHAP values (limit test set size for performance)
    test_sample = X_test.sample(min(50, len(X_test)), random_state=random_state)
    shap_values = explainer.shap_values(test_sample)
    
    # For binary classification, get positive class SHAP values
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    
    return explainer, shap_values, test_sample


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    output_path: Path,
    max_display: int = 20
) -> None:
    """Create SHAP summary plot."""
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, 
        X_sample, 
        max_display=max_display,
        show=False
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_shap_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    output_path: Path,
    max_display: int = 20
) -> None:
    """Create SHAP feature importance bar plot."""
    
    # Calculate mean absolute SHAP values
    importance = np.abs(shap_values).mean(axis=0)
    
    # Sort features by importance
    sorted_idx = np.argsort(importance)[-max_display:]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = importance[sorted_idx]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_importance)), sorted_importance)
    plt.yticks(range(len(sorted_importance)), sorted_features)
    plt.xlabel('Mean |SHAP Value|')
    plt.title('Feature Importance (SHAP)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_shap_waterfall_examples(
    explainer: shap.Explainer,
    X_sample: pd.DataFrame,
    output_dir: Path,
    gene_names: List[str] = None,
    n_examples: int = 3
) -> List[str]:
    """Create SHAP waterfall plots for example genes."""
    
    if gene_names is None:
        # Use first few samples as examples
        gene_names = [f"GENE{i+1}" for i in range(min(n_examples, len(X_sample)))]
    
    example_files = []
    
    for i, gene_name in enumerate(gene_names[:n_examples]):
        if i >= len(X_sample):
            break
            
        plt.figure(figsize=(10, 6))
        
        # Get SHAP values for this example
        shap_values_single = explainer.shap_values(X_sample.iloc[i:i+1])
        if isinstance(shap_values_single, list):
            shap_values_single = shap_values_single[1]  # Positive class
        
        # Create waterfall plot
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_single[0],
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
                data=X_sample.iloc[i].values,
                feature_names=list(X_sample.columns)
            ),
            show=False
        )
        
        plt.title(f'SHAP Explanation for {gene_name}')
        
        output_path = output_dir / f"example_{gene_name}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        example_files.append(str(output_path))
    
    return example_files


def generate_shap_explanations(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    output_dir: Path,
    model_type: str = "sklearn",
    random_state: int = 42
) -> Dict[str, Any]:
    """Generate comprehensive SHAP explanations."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute SHAP values
    explainer, shap_values, X_sample = compute_shap_values(
        model, X_train, X_test, model_type, random_state
    )
    
    # Generate plots
    shap_dir = output_dir / "shap"
    shap_dir.mkdir(exist_ok=True)
    
    # Global summary
    plot_shap_summary(shap_values, X_sample, shap_dir / "global_summary.png")
    
    # Feature importance
    plot_shap_feature_importance(
        shap_values, 
        list(X_sample.columns), 
        shap_dir / "feature_importance.png"
    )
    
    # Example waterfall plots
    example_files = plot_shap_waterfall_examples(
        explainer, X_sample, shap_dir, n_examples=3
    )
    
    # Calculate top features for each example
    top_features_per_example = []
    for i in range(min(3, len(X_sample))):
        if i < len(shap_values):
            feature_contributions = list(zip(X_sample.columns, shap_values[i]))
            # Sort by absolute SHAP value
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            top_features_per_example.append({
                f"GENE{i+1}": {
                    "top_positive": [(f, v) for f, v in feature_contributions if v > 0][:3],
                    "top_negative": [(f, v) for f, v in feature_contributions if v < 0][:3],
                    "prediction_score": float(X_sample.iloc[i].sum())  # Simplified score
                }
            })
    
    return {
        "shap_summary_path": str(shap_dir / "global_summary.png"),
        "example_paths": example_files,
        "top_features": top_features_per_example,
        "feature_importance": {
            "features": list(X_sample.columns),
            "importance": np.abs(shap_values).mean(axis=0).tolist()
        }
    }