"""PyTorch MLP training for toy immunotherapy target classification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class MLPConfig:
    hidden_sizes: tuple[int, ...] = (32, 16)
    dropout: float = 0.15
    lr: float = 1e-3
    epochs: int = 200
    patience: int = 20
    batch_size: int = 16
    validation_split: float = 0.2
    seed: int = 42


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, config: MLPConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_dim
        for hidden in config.hidden_sizes:
            layers.extend([
                nn.Linear(in_features, hidden),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ])
            in_features = hidden
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(inputs)


def _build_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    config: MLPConfig,
) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=config.validation_split,
        stratify=y,
        random_state=config.seed,
    )
    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    X_val = torch.from_numpy(X[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    train_ds = TensorDataset(X_train, y_train.unsqueeze(1))
    val_ds = TensorDataset(X_val, y_val.unsqueeze(1))
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    return train_loader, val_loader


def _evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            logits = model(batch_X.float())
            loss = criterion(logits, batch_y.float())
            losses.append(float(loss.detach().cpu().item()))
            probs = torch.sigmoid(logits)
            all_probs.append(probs.squeeze(1).cpu().numpy())
            all_targets.append(batch_y.squeeze(1).cpu().numpy())
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    y_prob = np.concatenate(all_probs) if all_probs else np.array([])
    y_true = np.concatenate(all_targets) if all_targets else np.array([])
    return mean_loss, y_prob, y_true


def train_mlp(
    features: pd.DataFrame,
    labels: pd.Series,
    config: MLPConfig | None = None,
    output_dir: Path | None = None,
) -> tuple[dict[str, float], MLPClassifier]:
    """Train the MLP and persist lightweight artifacts."""

    config = config or MLPConfig()
    output_dir = output_dir or (Path(__file__).resolve().parents[2] / "data" / "processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(config.seed)
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if not isinstance(labels, pd.Series):
        labels = pd.Series(labels, index=features.index)
    if not features.index.equals(labels.index):
        labels = labels.reindex(features.index)

    X = features.astype(np.float32).to_numpy()
    y = labels.astype(np.float32).to_numpy()

    train_loader, val_loader = _build_dataloaders(X, y, config)
    model = MLPClassifier(input_dim=X.shape[1], config=config)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X.float())
            loss = criterion(logits, batch_y.float())
            loss.backward()
            optimizer.step()
        val_loss, _, _ = _evaluate(model, val_loader, criterion)
        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_probs, val_true = _evaluate(model, val_loader, criterion)
    try:
        auroc = float(roc_auc_score(val_true, val_probs))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(val_true, val_probs))
    except ValueError:
        auprc = float("nan")

    metrics = {
        "val_loss": float(val_loss),
        "val_auroc": auroc,
        "val_auprc": auprc,
    }

    artifact = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "features": list(features.columns),
    }
    torch.save(artifact, output_dir / "mlp_model.pt")
    with (output_dir / "mlp_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics, model
