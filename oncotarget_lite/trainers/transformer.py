"""Transformer-based trainer for biological sequence data."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseTrainer, TrainerConfig, TrainingResult


class TransformerClassifier(nn.Module):
    """Transformer-based classifier for biological data."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_position_embeddings: int = 1000,
    ):
        super().__init__()

        # Project input features to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = nn.Embedding(max_position_embeddings, hidden_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape

        # Project input features
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)

        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_encoder(positions)

        # Apply transformer
        x = self.transformer_encoder(x)  # (batch_size, seq_len, hidden_dim)

        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, hidden_dim)

        # Apply dropout
        x = self.dropout(x)

        # Classification
        logits = self.classifier(x).squeeze(-1)  # (batch_size,)
        return torch.sigmoid(logits)


class TransformerTrainer(BaseTrainer):
    """Trainer for transformer-based models."""

    def __init__(self, config: TrainerConfig):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_pipeline(self) -> Pipeline:
        """Create sklearn pipeline wrapper for PyTorch model."""

        class TorchWrapper:
            def __init__(self, model, device):
                self.model = model
                self.device = device

            def fit(self, X, y):
                # Convert to tensors
                X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)  # Add sequence dimension
                y_tensor = torch.FloatTensor(y.values).to(self.device)

                # Create model
                input_dim = X.shape[1]
                model = TransformerClassifier(
                    input_dim=input_dim,
                    **self.config.model_params,
                ).to(self.device)

                # Training setup
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                criterion = nn.BCELoss()

                # Training loop
                model.train()
                for epoch in range(50):  # Simple training loop
                    optimizer.zero_grad()
                    outputs = model(X_tensor)
                    loss = criterion(outputs, y_tensor)
                    loss.backward()
                    optimizer.step()

                    if epoch % 10 == 0:
                        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

                self.model = model
                return self

            def predict_proba(self, X):
                self.model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)
                    probs = self.model(X_tensor).cpu().numpy()
                    return np.column_stack([1 - probs, probs])

            def predict(self, X):
                proba = self.predict_proba(X)
                return (proba[:, 1] > 0.5).astype(int)

        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", TorchWrapper(None, self.device)),
        ])
