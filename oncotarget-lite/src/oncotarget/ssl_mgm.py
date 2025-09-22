"""Self-supervised Masked Gene Modeling (MGM) pretraining for FT-Transformer."""

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

from .models_fttransformer import create_reconstruction_model
from .io import merge_gene_feature_table


def create_mask(
    x: torch.Tensor, 
    mask_frac: float = 0.15, 
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Create random mask for input features.
    
    Args:
        x: [batch_size, n_features] input tensor
        mask_frac: fraction of features to mask
        generator: torch generator for reproducibility
        
    Returns:
        mask: [batch_size, n_features] binary mask (1=masked)
    """
    batch_size, n_features = x.shape
    n_masked = int(n_features * mask_frac)
    
    # Create random masks for each sample
    masks = []
    for _ in range(batch_size):
        mask = torch.zeros(n_features, dtype=torch.bool)
        if n_masked > 0:
            indices = torch.randperm(n_features, generator=generator)[:n_masked]
            mask[indices] = True
        masks.append(mask)
    
    return torch.stack(masks).float()


def masked_mse_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    mask: torch.Tensor
) -> torch.Tensor:
    """Compute MSE loss only on masked positions.
    
    Args:
        pred: [batch_size, n_features] predicted values
        target: [batch_size, n_features] target values  
        mask: [batch_size, n_features] binary mask (1=compute loss)
        
    Returns:
        loss: scalar loss
    """
    masked_pred = pred[mask.bool()]
    masked_target = target[mask.bool()]
    
    if len(masked_pred) == 0:
        return torch.tensor(0.0, requires_grad=True)
        
    return nn.functional.mse_loss(masked_pred, masked_target)


def load_data_for_mgm(
    train_csv: Optional[str] = None,
    val_csv: Optional[str] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load data for MGM pretraining.
    
    Args:
        train_csv: path to training CSV (if None, use built-in data)
        val_csv: path to validation CSV (if None, use built-in data)
        
    Returns:
        X_train, X_val: tensors of continuous features
    """
    if train_csv is None or val_csv is None:
        # Use built-in synthetic data
        features_df = merge_gene_feature_table()
        
        # Select only continuous features (exclude boolean columns)
        continuous_cols = []
        for col in features_df.columns:
            if features_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                continuous_cols.append(col)
        
        features_df = features_df[continuous_cols]
        
        # Split into train/val
        n_total = len(features_df)
        n_train = int(0.8 * n_total)
        
        X_train = torch.from_numpy(features_df.iloc[:n_train].values.astype(np.float32))
        X_val = torch.from_numpy(features_df.iloc[n_train:].values.astype(np.float32))
        
    else:
        # Load from CSVs
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        
        # Assume first column is ID/label, rest are features
        X_train = torch.from_numpy(train_df.iloc[:, 1:].values.astype(np.float32))
        X_val = torch.from_numpy(val_df.iloc[:, 1:].values.astype(np.float32))
    
    return X_train, X_val


def train_mgm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    mask_frac: float = 0.15,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu"
) -> dict[str, list[float]]:
    """Train masked gene modeling.
    
    Args:
        model: FT-Transformer model for reconstruction
        train_loader: training data loader
        val_loader: validation data loader
        mask_frac: fraction of genes to mask
        epochs: number of training epochs
        lr: learning rate
        device: device to train on
        
    Returns:
        history: dict with train/val losses
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(42)
    
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch_x, in train_loader:
            batch_x = batch_x.to(device)
            
            # Create random mask
            mask = create_mask(batch_x, mask_frac, generator)
            mask = mask.to(device)
            
            # Forward pass
            pred = model(batch_x, mask)
            loss = masked_mse_loss(pred, batch_x, mask)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_x, in val_loader:
                batch_x = batch_x.to(device)
                mask = create_mask(batch_x, mask_frac, generator)
                mask = mask.to(device)
                
                pred = model(batch_x, mask)
                loss = masked_mse_loss(pred, batch_x, mask)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    return history


def main() -> None:
    """CLI for MGM pretraining."""
    parser = argparse.ArgumentParser(description="Masked Gene Modeling pretraining")
    
    parser.add_argument("--train-csv", type=str, help="Training CSV file")
    parser.add_argument("--val-csv", type=str, help="Validation CSV file") 
    parser.add_argument("--mask-frac", type=float, default=0.15, help="Fraction of genes to mask")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--save-encoder", type=str, default="encoder.pt", help="Path to save encoder weights")
    parser.add_argument("--save-model", type=str, help="Path to save full model (optional)")
    parser.add_argument("--save-history", type=str, help="Path to save training history (optional)")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    X_train, X_val = load_data_for_mgm(args.train_csv, args.val_csv)
    n_features = X_train.shape[1]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Number of features: {n_features}")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train)
    val_dataset = TensorDataset(X_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print("Creating model...")
    model = create_reconstruction_model(
        n_features=n_features,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("Starting training...")
    history = train_mgm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mask_frac=args.mask_frac,
        epochs=args.epochs,
        lr=args.lr
    )
    
    # Save encoder weights
    print(f"Saving encoder weights to {args.save_encoder}")
    encoder_state = model.get_encoder_state_dict()
    torch.save(encoder_state, args.save_encoder)
    
    # Save full model if requested
    if args.save_model:
        print(f"Saving full model to {args.save_model}")
        torch.save({
            "state_dict": model.state_dict(),
            "n_features": n_features,
            "config": {
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "task": "reconstruction"
            }
        }, args.save_model)
    
    # Save training history if requested  
    if args.save_history:
        print(f"Saving training history to {args.save_history}")
        with open(args.save_history, "w") as f:
            json.dump(history, f, indent=2)
    
    print("MGM pretraining complete!")


if __name__ == "__main__":
    main()