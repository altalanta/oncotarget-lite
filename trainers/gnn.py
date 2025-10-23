"""Graph Neural Network trainer for biological network data."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseTrainer, TrainerConfig, TrainingResult


class GNNClassifier(nn.Module):
    """Graph Neural Network classifier for biological networks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_heads: int = 8,
    ):
        super().__init__()

        # Node embedding layers
        self.node_embeddings = nn.ModuleList()
        self.node_embeddings.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.node_embeddings.append(nn.Linear(hidden_dim, hidden_dim))

        # Attention mechanism for aggregating neighbor information
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Graph-level readout
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for graph neural network.

        Args:
            x: Node features of shape (batch_size, num_nodes, input_dim)

        Returns:
            Graph-level predictions of shape (batch_size,)
        """
        batch_size, num_nodes, input_dim = x.shape

        # Apply node embedding layers
        node_features = x
        for layer in self.node_embeddings:
            node_features = F.relu(layer(node_features))
            node_features = F.dropout(node_features, training=self.training)

        # For simplicity, we'll treat each feature vector as a "graph"
        # In a real implementation, you'd have actual graph structure

        # Global pooling (mean pooling across nodes)
        graph_features = node_features.mean(dim=1)  # (batch_size, hidden_dim)

        # Apply global pooling layers
        graph_features = self.global_pool(graph_features)

        # Final classification
        logits = self.classifier(graph_features).squeeze(-1)
        return torch.sigmoid(logits)


class GNNTrainer(BaseTrainer):
    """Trainer for Graph Neural Network models."""

    def __init__(self, config: TrainerConfig):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_pipeline(self) -> Pipeline:
        """Create sklearn pipeline wrapper for PyTorch GNN model."""

        class TorchGNNWrapper:
            def __init__(self, model, device):
                self.model = model
                self.device = device

            def fit(self, X, y):
                # Convert to tensors - reshape to simulate graph structure
                # In practice, you'd have actual graph data
                X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)  # Add "node" dimension
                y_tensor = torch.FloatTensor(y.values).to(self.device)

                # Create model
                input_dim = X.shape[1]
                model = GNNClassifier(
                    input_dim=input_dim,
                    **self.config.model_params,
                ).to(self.device)

                # Training setup
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                criterion = nn.BCELoss()

                # Training loop
                model.train()
                for epoch in range(100):  # More epochs for GNN
                    optimizer.zero_grad()
                    outputs = model(X_tensor)
                    loss = criterion(outputs, y_tensor)
                    loss.backward()
                    optimizer.step()

                    if epoch % 20 == 0:
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
            ("classifier", TorchGNNWrapper(None, self.device)),
        ])
