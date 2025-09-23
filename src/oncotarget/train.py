"""CLI for training models with support for MLP and FT-Transformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .io import merge_gene_feature_table
from .model import MLPClassifier, MLPConfig
from .models_fttransformer import create_fttransformer_for_task


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data_from_csv(
    train_csv: str,
    val_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    target_col: str = "target",
    validation_split: float = 0.2,
    seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load data from CSV files.
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Load training data
    train_df = pd.read_csv(train_csv)
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {train_csv}")
    
    X_train_df = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values
    
    # Load validation data or split from training
    if val_csv:
        val_df = pd.read_csv(val_csv)
        X_val_df = val_df.drop(columns=[target_col])
        y_val = val_df[target_col].values
        X_train = X_train_df.values
    else:
        # Split training data
        indices = np.arange(len(X_train_df))
        train_idx, val_idx = train_test_split(
            indices, test_size=validation_split, stratify=y_train, random_state=seed
        )
        X_train = X_train_df.iloc[train_idx].values
        y_train = y_train[train_idx]
        X_val_df = X_train_df.iloc[val_idx]
        X_val = X_val_df.values
        y_val = y_train[val_idx]
    
    # Load test data if provided
    X_test, y_test = None, None
    if test_csv:
        test_df = pd.read_csv(test_csv)
        X_test = test_df.drop(columns=[target_col]).values
        y_test = test_df[target_col].values
    
    # Convert to tensors
    X_train = torch.from_numpy(X_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    X_val = torch.from_numpy(X_val.astype(np.float32))
    y_val = torch.from_numpy(y_val.astype(np.float32))
    
    if X_test is not None:
        X_test = torch.from_numpy(X_test.astype(np.float32))
        y_test = torch.from_numpy(y_test.astype(np.float32))
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_built_in_data(validation_split: float = 0.2, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load built-in synthetic data."""
    features_df = merge_gene_feature_table()
    
    # Create synthetic binary labels (cell surface proteins are positive)
    if 'is_cell_surface' in features_df.columns:
        labels = features_df['is_cell_surface'].astype(float)
        features_df = features_df.drop(columns=['is_cell_surface'])
    else:
        # Random labels for demo
        np.random.seed(seed)
        labels = pd.Series(np.random.binomial(1, 0.3, len(features_df)), index=features_df.index)
    
    # Select only continuous features
    continuous_cols = []
    for col in features_df.columns:
        if features_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            continuous_cols.append(col)
    
    features_df = features_df[continuous_cols]
    
    # Split data
    indices = np.arange(len(features_df))
    train_idx, val_idx = train_test_split(
        indices, test_size=validation_split, stratify=labels, random_state=seed
    )
    
    X_train = torch.from_numpy(features_df.iloc[train_idx].values.astype(np.float32))
    y_train = torch.from_numpy(labels.iloc[train_idx].values.astype(np.float32))
    X_val = torch.from_numpy(features_df.iloc[val_idx].values.astype(np.float32))
    y_val = torch.from_numpy(labels.iloc[val_idx].values.astype(np.float32))
    
    return X_train, y_train, X_val, y_val


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    task: str = "binary"
) -> dict[str, float]:
    """Evaluate model and return metrics."""
    model.eval()
    losses = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            outputs = model(batch_x)
            # Handle different model output shapes
            if outputs.dim() > 1:
                outputs = outputs.squeeze(-1)
            loss = criterion(outputs, batch_y)
            losses.append(loss.item())
            
            if task == "binary":
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
            else:  # regression
                all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    avg_loss = np.mean(losses)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    
    metrics = {"loss": avg_loss}
    
    if task == "binary":
        try:
            metrics["auroc"] = roc_auc_score(y_true, y_pred)
            metrics["auprc"] = average_precision_score(y_true, y_pred)
        except ValueError:
            metrics["auroc"] = float("nan")
            metrics["auprc"] = float("nan")
    else:  # regression
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["r2"] = r2_score(y_true, y_pred)
    
    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task: str = "binary",
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 20
) -> tuple[dict[str, float], nn.Module]:
    """Train model with early stopping."""
    
    if task == "binary":
        criterion = nn.BCEWithLogitsLoss()
    else:  # regression
        criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_x)
            if task == "binary":
                # Handle different model output shapes
                if outputs.dim() > 1:
                    outputs = outputs.squeeze(-1)
                loss = criterion(outputs, batch_y)
            else:
                if outputs.dim() > 1:
                    outputs = outputs.squeeze(-1)
                loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, criterion, task)
        val_loss = val_metrics["loss"]
        
        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: val_loss={val_loss:.4f}")
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Final evaluation
    final_metrics = evaluate_model(model, val_loader, criterion, task)
    return final_metrics, model


def main() -> None:
    """CLI for model training."""
    parser = argparse.ArgumentParser(description="Train models for oncotarget prediction")
    
    # Data arguments
    parser.add_argument("--train-csv", type=str, help="Training CSV file")
    parser.add_argument("--val-csv", type=str, help="Validation CSV file") 
    parser.add_argument("--test-csv", type=str, help="Test CSV file")
    parser.add_argument("--target-col", type=str, default="target", help="Target column name")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split if no val-csv")
    
    # Model arguments
    parser.add_argument("--model", type=str, choices=["mlp", "fttransformer"], default="mlp", help="Model type")
    parser.add_argument("--task", type=str, choices=["binary", "regression"], default="binary", help="Task type")
    parser.add_argument("--pretrained-encoder", type=str, help="Path to pretrained encoder weights")
    
    # MLP arguments
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[32, 16], help="MLP hidden layer sizes")
    parser.add_argument("--mlp-dropout", type=float, default=0.15, help="MLP dropout")
    
    # FT-Transformer arguments
    parser.add_argument("--d-model", type=int, default=128, help="Transformer model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output directory")
    parser.add_argument("--model-name", type=str, help="Model filename (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Set seed
    _set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    if args.train_csv:
        X_train, y_train, X_val, y_val, X_test, y_test = load_data_from_csv(
            args.train_csv, args.val_csv, args.test_csv, args.target_col, args.validation_split, args.seed
        )
    else:
        print("Using built-in synthetic data...")
        X_train, y_train, X_val, y_val = load_built_in_data(args.validation_split, args.seed)
        X_test, y_test = None, None
    
    n_features = X_train.shape[1]
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {n_features}")
    if X_test is not None:
        print(f"Test samples: {len(X_test)}")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print(f"Creating {args.model} model...")
    if args.model == "mlp":
        config = MLPConfig(
            hidden_sizes=tuple(args.hidden_sizes),
            dropout=args.mlp_dropout,
            lr=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            seed=args.seed
        )
        model = MLPClassifier(input_dim=n_features, config=config)
    else:  # fttransformer
        model = create_fttransformer_for_task(
            n_features=n_features,
            task=args.task,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout
        )
        
        # Load pretrained encoder if provided
        if args.pretrained_encoder:
            print(f"Loading pretrained encoder from {args.pretrained_encoder}")
            encoder_state = torch.load(args.pretrained_encoder, map_location="cpu")
            model.load_encoder_state_dict(encoder_state)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("Starting training...")
    metrics, trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task=args.task,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience
    )
    
    print("Validation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Evaluate on test set if available
    if X_test is not None:
        print("Evaluating on test set...")
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        if args.task == "binary":
            test_criterion = nn.BCEWithLogitsLoss()
        else:
            test_criterion = nn.MSELoss()
            
        test_metrics = evaluate_model(trained_model, test_loader, test_criterion, args.task)
        print("Test metrics:")
        for name, value in test_metrics.items():
            print(f"  {name}: {value:.4f}")
        metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
    
    # Save model and metrics
    model_name = args.model_name or f"{args.model}_{args.task}_model.pt"
    model_path = output_dir / model_name
    metrics_path = output_dir / f"{model_name.replace('.pt', '_metrics.json')}"
    
    print(f"Saving model to {model_path}")
    torch.save({
        "state_dict": trained_model.state_dict(),
        "model_type": args.model,
        "task": args.task,
        "n_features": n_features,
        "config": vars(args)
    }, model_path)
    
    print(f"Saving metrics to {metrics_path}")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("Training complete!")


if __name__ == "__main__":
    main()