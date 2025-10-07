"""SHAP-based interpretability utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .data import PROCESSED_DIR
from .model import MODELS_DIR
from .utils import ensure_dir, load_json, set_seeds

SHAP_DIR = Path("reports/shap")
EXAMPLE_ALIASES = ["GENE1", "GENE2", "GENE3"]


@dataclass(slots=True)
class ShapArtifacts:
    global_importance: pd.Series
    per_gene_contribs: dict[str, pd.Series]
    alias_map: dict[str, str]


class ExplanationError(RuntimeError):
    """Raised when SHAP explanations cannot be computed."""


def _load_training_state(processed_dir: Path, models_dir: Path):
    from joblib import load

    features = pd.read_parquet(processed_dir / "features.parquet")
    splits = load_json(processed_dir / "splits.json")
    model_path = models_dir / "logreg_pipeline.pkl"
    if not model_path.exists():
        raise ExplanationError("Trained model not found; run train before explain")
    pipeline = load(model_path)
    return features, splits, pipeline


def _select_examples(test_genes: Sequence[str], *, k: int = 3) -> list[str]:
    return list(test_genes)[:k]


def _plot_global(feature_names: list[str], importances: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    order = np.argsort(importances)
    sorted_features = [feature_names[i] for i in order]
    sorted_importances = importances[order]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(sorted_features, sorted_importances, color="#4c72b0")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_gene(gene: str, alias: str, contributions: pd.Series, path: Path) -> None:
    ensure_dir(path.parent)
    sorted_contribs = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
    top = sorted_contribs.head(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#dd8452" if val >= 0 else "#55a868" for val in top]
    ax.barh(top.index, top.values, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP contribution")
    ax.set_title(f"{gene} ({alias}) feature contributions")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_shap(
    *,
    processed_dir: Path = PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
    shap_dir: Path = SHAP_DIR,
    seed: int = 42,
    background_size: int = 100,
    distributed: bool = True,
    max_evals: Optional[int] = None,
) -> ShapArtifacts:
    """Compute SHAP values for the trained model and persist plots."""

    # Configure distributed computing if enabled
    if distributed:
        from .distributed import configure_distributed
        configure_distributed(backend='joblib', n_jobs=-1, verbose=0)

    set_seeds(seed)
    features, splits, pipeline = _load_training_state(processed_dir, models_dir)
    train_genes: list[str] = splits["train_genes"]
    test_genes: list[str] = splits["test_genes"]

    if not train_genes or not test_genes:
        raise ExplanationError("Missing train/test splits for SHAP")

    background = features.loc[train_genes]
    if len(background) > background_size:
        background = background.sample(background_size, random_state=seed)

    target = features.loc[test_genes]

    def predict_fn(data: pd.DataFrame) -> np.ndarray:
        preds = pipeline.predict_proba(data)[:, 1]
        return preds

    explainer = shap.Explainer(predict_fn, background, seed=seed)
    shap_values = explainer(target)
    values = np.array(shap_values.values)
    if values.ndim == 2:
        per_gene_values = values
    else:  # KernelExplainer may return shape (n_samples, 1, n_features)
        per_gene_values = values.reshape(values.shape[0], -1)

    mean_abs = np.mean(np.abs(per_gene_values), axis=0)
    feature_names = list(features.columns)
    _plot_global(feature_names, mean_abs, shap_dir / "global_summary.png")

    alias_map: dict[str, str] = {}
    per_gene_contribs: dict[str, pd.Series] = {}

    example_genes = _select_examples(test_genes, k=len(EXAMPLE_ALIASES))
    for gene, alias in zip(example_genes, EXAMPLE_ALIASES):
        idx = target.index.get_loc(gene)
        contrib = pd.Series(per_gene_values[idx], index=feature_names)
        per_gene_contribs[gene] = contrib
        alias_map[alias] = gene
        _plot_gene(
            gene,
            alias,
            contrib,
            shap_dir / f"example_{alias}.png",
        )

    shap_dir.mkdir(parents=True, exist_ok=True)
    shap_values_path = shap_dir / "shap_values.npz"
    np.savez(
        shap_values_path,
        genes=np.array(target.index, dtype=object),
        values=per_gene_values,
        feature_names=np.array(feature_names, dtype=object),
    )
    alias_path = shap_dir / "alias_map.json"
    alias_path.write_text(json.dumps(alias_map, indent=2), encoding="utf-8")

    return ShapArtifacts(
        global_importance=pd.Series(mean_abs, index=feature_names).sort_values(ascending=False),
        per_gene_contribs=per_gene_contribs,
        alias_map=alias_map,
    )
