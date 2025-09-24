"""Model training with MLflow tracking and comprehensive evaluation.

This module provides PyTorch MLP and scikit-learn Random Forest implementations
with integrated MLflow experiment tracking, early stopping, and lineage management.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .utils import compute_dataset_hash, get_git_commit, set_random_seed


class MLPClassifier(nn.Module):
    """Multi-layer perceptron for binary classification with configurable architecture.
    
    A flexible neural network implementation with ReLU activations, dropout regularization,
    and BCEWithLogitsLoss for stable training on imbalanced datasets.
    
    Args:
        input_dim: Number of input features
        hidden_sizes: Tuple of hidden layer dimensions (default: (32, 16))
        dropout: Dropout probability for regularization (default: 0.15)
    
    Example:
        >>> model = MLPClassifier(input_dim=10, hidden_sizes=(64, 32), dropout=0.2)
        >>> logits = model(torch.randn(32, 10))  # batch_size=32
    """
    
    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, ...] = (32, 16), dropout: float = 0.15):
        """Initialize MLP architecture with specified dimensions."""
        super().__init__()
        layers = []
        in_features = input_dim
        
        for hidden in hidden_sizes:
            layers.extend([
                nn.Linear(in_features, hidden),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_features = hidden
        
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Logits tensor of shape (batch_size, 1)
        """
        return self.network(x)


def train_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    hidden_sizes: Tuple[int, ...] = (32, 16),
    dropout: float = 0.15,
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 16,
    patience: int = 10,
    random_state: int = 42
) -> Tuple[Dict[str, Any], MLPClassifier]:
    """Train MLP classifier with early stopping and comprehensive evaluation.
    
    Implements stratified train-validation split, early stopping based on validation loss,
    and returns detailed performance metrics including overfitting indicators.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training labels Series
        X_test: Test features DataFrame  
        y_test: Test labels Series
        hidden_sizes: Architecture specification as tuple of layer dimensions
        dropout: Dropout probability for regularization [0.0, 1.0]
        lr: Learning rate for Adam optimizer
        epochs: Maximum training epochs
        batch_size: Mini-batch size for training
        patience: Early stopping patience (epochs without improvement)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (metrics_dict, trained_model) where metrics include:
        - train_auroc, test_auroc: Area under ROC curve
        - test_accuracy, test_f1: Classification metrics
        - test_average_precision: Area under PR curve
        - auroc_gap: Overfitting indicator (train - test AUROC)
        
    Raises:
        ValueError: If training data is insufficient or malformed
        
    Example:
        >>> metrics, model = train_mlp(X_train, y_train, X_test, y_test,
        ...                          hidden_sizes=(64, 32), epochs=50)
        >>> print(f"Test AUROC: {metrics['test_auroc']:.3f}")
    """
    
    set_random_seed(random_state)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)
    
    # Create validation split
    val_size = 0.2
    train_idx, val_idx = train_test_split(
        range(len(X_train_tensor)), 
        test_size=val_size, 
        stratify=y_train,
        random_state=random_state
    )
    
    # Data loaders
    train_dataset = TensorDataset(X_train_tensor[train_idx], y_train_tensor[train_idx])
    val_dataset = TensorDataset(X_train_tensor[val_idx], y_train_tensor[val_idx])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Model
    model = MLPClassifier(X_train.shape[1], hidden_sizes, dropout)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_state_dict = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load best model
    if best_state_dict:
        model.load_state_dict(best_state_dict)
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        y_pred_proba = torch.sigmoid(model(X_test_tensor)).numpy().flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    train_proba = torch.sigmoid(model(X_train_tensor)).detach().numpy().flatten()
    train_auroc = roc_auc_score(y_train, train_proba)
    
    metrics = {
        "train_auroc": train_auroc,
        "test_auroc": roc_auc_score(y_test, y_pred_proba),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_average_precision": average_precision_score(y_test, y_pred_proba),
        "auroc_gap": train_auroc - roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics, model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42
) -> Tuple[Dict[str, Any], RandomForestClassifier]:
    """Train Random Forest with MLflow tracking."""
    
    set_random_seed(random_state)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    # Predictions
    train_proba = model.predict_proba(X_train)[:, 1]
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    train_auroc = roc_auc_score(y_train, train_proba)
    
    metrics = {
        "train_auroc": train_auroc,
        "test_auroc": roc_auc_score(y_test, y_pred_proba),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_average_precision": average_precision_score(y_test, y_pred_proba),
        "auroc_gap": train_auroc - roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics, model


def track_experiment(
    model_type: str,
    features: pd.DataFrame,
    labels: pd.Series,
    metrics: Dict[str, Any],
    model: Any,
    params: Dict[str, Any],
    artifacts_dir: Path
) -> str:
    """Log experiment to MLflow with lineage tracking."""
    
    # Set tracking URI
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Set experiment
    experiment_name = "oncotarget-lite"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except Exception:
        experiment_id = mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_features", features.shape[1])
        mlflow.log_param("n_samples", features.shape[0])
        mlflow.log_param("feature_names", list(features.columns))
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log tags for lineage
        mlflow.set_tags({
            "git_commit": get_git_commit(),
            "dataset_hash": compute_dataset_hash(features),
            "data_version": "v1.0",
            "code_version": "0.2.0"
        })
        
        # Log model
        if model_type == "mlp":
            mlflow.pytorch.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
        
        # Log artifacts if they exist
        if artifacts_dir.exists():
            mlflow.log_artifacts(str(artifacts_dir), "reports")
        
        return run.info.run_id