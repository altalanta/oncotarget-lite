"""Neural network architectures."""

from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Small feed-forward network for binary classification."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.network(x)
        return logits.squeeze(-1)


def compute_feature_importance(model: MLPClassifier, feature_names: list[str]) -> dict[str, float]:
    """Estimate feature importance from the first linear layer weights."""

    first_linear = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            first_linear = module
            break
    if first_linear is None:
        raise RuntimeError("MLPClassifier missing Linear layer")
    weights = first_linear.weight.detach().abs().mean(dim=0)
    importances = {
        name: float(weight)
        for name, weight in zip(feature_names, weights.tolist(), strict=True)
    }
    return importances
