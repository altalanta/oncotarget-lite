"""Training orchestration for the oncotarget-lite MLP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import EvaluationConfig, SplitConfig, TrainingConfig
from .data import DataContractError, split_dataset
from .features import FeatureSet
from .metrics import (
    BootstrapSummary,
    MetricSummary,
    auprc,
    auroc,
    bootstrap_confidence_interval,
    brier_score,
    summarise_metrics,
)
from .model import MLPClassifier, compute_feature_importance
from .utils import select_device, set_global_seed

MIN_TRAIN_SAMPLES = 4
VAL_FRACTION = 0.2


@dataclass(frozen=True)
class TrainingArtifacts:
    model: MLPClassifier
    device: torch.device
    train_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    feature_importances: pd.Series
    train_metrics: MetricSummary
    test_metrics: MetricSummary
    bootstrap: BootstrapSummary
    calibration: pd.DataFrame


def _build_loaders(
    features: pd.DataFrame,
    labels: pd.Series,
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    indices = np.arange(len(features))
    if len(indices) < MIN_TRAIN_SAMPLES:
        msg = "training set too small to form validation split"
        raise DataContractError(msg)
    rng = np.random.default_rng(config.seed)
    perm = rng.permutation(indices)
    val_size = max(1, int(round(len(indices) * VAL_FRACTION)))
    if val_size >= len(indices):
        val_size = len(indices) - 1
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    X_train = torch.from_numpy(features.to_numpy(dtype=np.float32)[train_idx])
    y_train = torch.from_numpy(labels.to_numpy(dtype=np.float32)[train_idx])
    X_val = torch.from_numpy(features.to_numpy(dtype=np.float32)[val_idx])
    y_val = torch.from_numpy(labels.to_numpy(dtype=np.float32)[val_idx])

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    return train_loader, val_loader


def _predict(
    model: MLPClassifier, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for features_batch, targets_batch in loader:
            logits = model(features_batch.to(device))
            prob = sigmoid(logits).cpu().numpy()
            probs.append(prob)
            targets.append(targets_batch.cpu().numpy())
    return np.concatenate(probs), np.concatenate(targets)


def train_pipeline(
    feature_set: FeatureSet,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    eval_cfg: EvaluationConfig,
    output_dir: str | None = None,
) -> TrainingArtifacts:
    """Run deterministic train/test split, train the MLP, and collect metrics."""

    set_global_seed(train_cfg.seed)
    splits = split_dataset(feature_set.features, feature_set.labels, split_cfg)
    device = select_device(train_cfg.device)

    train_loader, val_loader = _build_loaders(splits.X_train, splits.y_train, train_cfg)
    model = MLPClassifier(
        input_dim=splits.X_train.shape[1],
        hidden_dims=train_cfg.hidden_dims,
        dropout=train_cfg.dropout,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
    )

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0

    for _epoch in range(train_cfg.max_epochs):
        model.train()
        for features_batch, targets_batch in train_loader:
            optimizer.zero_grad()
            logits = model(features_batch.to(device))
            loss = criterion(logits, targets_batch.to(device))
            loss.backward()
            optimizer.step()
        val_probs, val_true = _predict(model, val_loader, device)
        val_loss = float(np.mean((val_probs - val_true) ** 2))
        if val_loss + 1e-6 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Collect predictions on train/test splits
    train_dataset = TensorDataset(
        torch.from_numpy(splits.X_train.to_numpy(dtype=np.float32)),
        torch.from_numpy(splits.y_train.to_numpy(dtype=np.float32)),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(splits.X_test.to_numpy(dtype=np.float32)),
        torch.from_numpy(splits.y_test.to_numpy(dtype=np.float32)),
    )
    train_loader_full = DataLoader(train_dataset, batch_size=train_cfg.batch_size)
    test_loader_full = DataLoader(test_dataset, batch_size=train_cfg.batch_size)

    train_probs, train_true = _predict(model, train_loader_full, device)
    test_probs, test_true = _predict(model, test_loader_full, device)

    test_metrics, calib_curve = summarise_metrics(test_true, test_probs, eval_cfg.calibration_bins)
    train_metrics, _ = summarise_metrics(train_true, train_probs, eval_cfg.calibration_bins)

    bootstrap_summary = BootstrapSummary(
        auroc=bootstrap_confidence_interval(
            test_true,
            test_probs,
            auroc,
            eval_cfg.bootstrap_samples,
            eval_cfg.ci_level,
            eval_cfg.seed,
        ),
        auprc=bootstrap_confidence_interval(
            test_true,
            test_probs,
            auprc,
            eval_cfg.bootstrap_samples,
            eval_cfg.ci_level,
            eval_cfg.seed,
        ),
        brier=bootstrap_confidence_interval(
            test_true,
            test_probs,
            brier_score,
            eval_cfg.bootstrap_samples,
            eval_cfg.ci_level,
            eval_cfg.seed,
        ),
    )

    train_predictions = pd.DataFrame(
        {
            "gene": splits.X_train.index,
            "split": "train",
            "y_true": train_true,
            "y_prob": train_probs,
        }
    ).set_index("gene")
    test_predictions = pd.DataFrame(
        {
            "gene": splits.X_test.index,
            "split": "test",
            "y_true": test_true,
            "y_prob": test_probs,
        }
    ).set_index("gene")

    feature_importances = pd.Series(
        compute_feature_importance(model, list(splits.X_train.columns)),
        name="importance",
    ).sort_values(ascending=False)

    return TrainingArtifacts(
        model=model,
        device=device,
        train_predictions=train_predictions,
        test_predictions=test_predictions,
        feature_importances=feature_importances,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        bootstrap=bootstrap_summary,
        calibration=calib_curve,
    )
