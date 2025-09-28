"""Reporting utilities for oncotarget-lite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import ensure_dir, load_dataframe, load_json, write_text

SCORECARD_PATH = Path("reports/target_scorecard.html")
DOCS_INDEX = Path("docs/index.html")
RUN_CONTEXT = Path("reports/run_context.json")
README_PATH = Path("README.md")


class ReportingError(RuntimeError):
    """Raised when report artefacts cannot be generated."""


def _load_metrics(reports_dir: Path) -> dict[str, Any]:
    metrics_path = reports_dir / "metrics.json"
    if not metrics_path.exists():
        raise ReportingError("metrics.json not found; run eval before scorecard/docs")
    return load_json(metrics_path)


def _load_bootstrap(reports_dir: Path) -> dict[str, Any]:
    bootstrap_path = reports_dir / "bootstrap.json"
    if not bootstrap_path.exists():
        raise ReportingError("bootstrap.json not found; run eval before scorecard/docs")
    return load_json(bootstrap_path)


def _load_predictions(reports_dir: Path) -> pd.DataFrame:
    preds_path = reports_dir / "predictions.parquet"
    if not preds_path.exists():
        raise ReportingError("predictions.parquet missing; run train before scorecard")
    return load_dataframe(preds_path)


def _rank_test_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    test_preds = preds[preds["split"] == "test"].copy()
    if test_preds.empty:
        raise ReportingError("No test predictions available for scorecard")
    test_preds.sort_values("y_prob", ascending=False, inplace=True)
    test_preds["rank"] = np.arange(1, len(test_preds) + 1)
    test_preds["percentile"] = 1 - (test_preds["rank"] - 0.5) / len(test_preds)
    return test_preds.set_index("gene")


def _load_shap_arrays(shap_dir: Path) -> tuple[dict[str, str], dict[str, np.ndarray], list[str]]:
    alias_path = shap_dir / "alias_map.json"
    shap_npz = shap_dir / "shap_values.npz"
    if not alias_path.exists() or not shap_npz.exists():
        raise ReportingError("SHAP artefacts missing; run explain before scorecard")
    alias_map = json.loads(alias_path.read_text(encoding="utf-8"))
    payload = np.load(shap_npz, allow_pickle=True)
    genes = payload["genes"].tolist()
    values = payload["values"]
    feature_names = payload["feature_names"].tolist()
    shap_lookup = {gene: values[idx] for idx, gene in enumerate(genes)}
    return alias_map, shap_lookup, feature_names


def _list_items(series: pd.Series) -> str:
    return "".join(f"<li><strong>{feat}</strong>: {val:+.3f}</li>" for feat, val in series.items())


def _describe_gene(
    gene: str,
    alias: str,
    ranked_preds: pd.DataFrame,
    shap_lookup: dict[str, np.ndarray],
    feature_names: list[str],
) -> str:
    if gene not in ranked_preds.index:
        raise ReportingError(f"Gene {gene} missing from predictions")
    if gene not in shap_lookup:
        raise ReportingError(f"Gene {gene} missing from SHAP values")
    stats = ranked_preds.loc[gene]
    contribs = pd.Series(shap_lookup[gene], index=feature_names)
    top_pos = contribs.sort_values(ascending=False).head(3)
    top_neg = contribs.sort_values().head(3)
    shap_img = f"shap/example_{alias}.png"
    return f"""
    <div class="card">
      <h3>{gene} ({alias})</h3>
      <p>Predicted score: <strong>{stats['y_prob']:.3f}</strong> · Rank: {int(stats['rank'])} / {len(ranked_preds)} · Percentile: {stats['percentile']*100:.1f}%</p>
      <div class="card-body">
        <div>
          <h4>Top positive contributors</h4>
          <ul>{_list_items(top_pos)}</ul>
        </div>
        <div>
          <h4>Top negative contributors</h4>
          <ul>{_list_items(top_neg)}</ul>
        </div>
      </div>
      <p><a href="{shap_img}"><img src="{shap_img}" alt="SHAP {gene}" /></a></p>
    </div>
    """


def _metric_table_html(metrics: dict[str, Any], bootstrap: dict[str, Any]) -> str:
    rows = [
        ("AUROC", f"{metrics['auroc']:.3f}", f"{bootstrap['auroc']['lower']:.3f} – {bootstrap['auroc']['upper']:.3f}"),
        ("Average Precision", f"{metrics['ap']:.3f}", f"{bootstrap['ap']['lower']:.3f} – {bootstrap['ap']['upper']:.3f}"),
        ("Brier", f"{metrics['brier']:.3f}", "–"),
        ("ECE", f"{metrics['ece']:.3f}", "–"),
        ("Accuracy", f"{metrics['accuracy']:.3f}", "–"),
        ("F1", f"{metrics['f1']:.3f}", "–"),
        ("Train AUROC", f"{metrics['train_auroc']:.3f}", "–"),
        ("Test AUROC", f"{metrics['test_auroc']:.3f}", "–"),
        ("Overfit gap", f"{metrics['overfit_gap']:.3f}", "–"),
    ]
    body = "".join(
        f"<tr><td>{name}</td><td>{value}</td><td>{ci}</td></tr>" for name, value, ci in rows
    )
    return f"<table><tr><th>Metric</th><th>Value</th><th>95% CI</th></tr>{body}</table>"


def _metric_table_markdown(metrics: dict[str, Any], bootstrap: dict[str, Any]) -> str:
    rows = [
        ("AUROC", f"{metrics['auroc']:.3f}", f"{bootstrap['auroc']['lower']:.3f} – {bootstrap['auroc']['upper']:.3f}"),
        ("Average Precision", f"{metrics['ap']:.3f}", f"{bootstrap['ap']['lower']:.3f} – {bootstrap['ap']['upper']:.3f}"),
        ("Brier", f"{metrics['brier']:.3f}", "–"),
        ("ECE", f"{metrics['ece']:.3f}", "–"),
        ("Accuracy", f"{metrics['accuracy']:.3f}", "–"),
        ("F1", f"{metrics['f1']:.3f}", "–"),
        ("Train AUROC", f"{metrics['train_auroc']:.3f}", "–"),
        ("Test AUROC", f"{metrics['test_auroc']:.3f}", "–"),
        ("Overfit gap", f"{metrics['overfit_gap']:.3f}", "–"),
    ]
    header = "| Metric | Value | 95% CI |\n| --- | --- | --- |"
    body = "\n".join(f"| {name} | {value} | {ci} |" for name, value, ci in rows)
    return f"{header}\n{body}"


def _update_model_card(model_card: Path, table: str) -> None:
    if not model_card.exists():
        return
    content = model_card.read_text(encoding="utf-8")
    start_marker = "<!-- METRICS_TABLE_START -->"
    end_marker = "<!-- METRICS_TABLE_END -->"
    if start_marker not in content or end_marker not in content:
        return
    new_section = f"{start_marker}\n{table}\n{end_marker}"
    parts = content.split(start_marker)
    suffix = parts[1].split(end_marker)[1]
    updated = parts[0] + new_section + suffix
    model_card.write_text(updated, encoding="utf-8")


def _update_readme(table: str) -> None:
    if not README_PATH.exists():
        return
    content = README_PATH.read_text(encoding="utf-8")
    start_marker = "<!-- README_METRICS_START -->"
    end_marker = "<!-- README_METRICS_END -->"
    if start_marker not in content or end_marker not in content:
        return
    prefix, remainder = content.split(start_marker, 1)
    _, suffix = remainder.split(end_marker, 1)
    new_section = f"{start_marker}\n{table}\n{end_marker}"
    README_PATH.write_text(prefix + new_section + suffix, encoding="utf-8")


def _mlflow_link() -> str | None:
    if not RUN_CONTEXT.exists():
        return None
    ctx = json.loads(RUN_CONTEXT.read_text(encoding="utf-8"))
    run_id = ctx.get("run_id")
    if not run_id:
        return None
    return f"mlflow://runs/{run_id}"


def generate_scorecard(
    *,
    reports_dir: Path = Path("reports"),
    shap_dir: Path = Path("reports/shap"),
    output_path: Path = SCORECARD_PATH,
) -> Path:
    """Compose a compact HTML scorecard linking metrics, ranks, and SHAP artefacts."""

    metrics = _load_metrics(reports_dir)
    bootstrap = _load_bootstrap(reports_dir)
    preds = _load_predictions(reports_dir)
    ranked = _rank_test_predictions(preds)
    alias_map, shap_lookup, feature_names = _load_shap_arrays(shap_dir)

    html = [
        "<html><head><meta charset='utf-8'><title>oncotarget-lite scorecard</title>",
        "<style>body{font-family:Arial,sans-serif;margin:2rem;}h1{margin-bottom:0.5rem;}table{border-collapse:collapse;margin-top:1rem;}td,th{padding:0.4rem 0.8rem;border:1px solid #ccc;}div.card{border:1px solid #d0d0d0;padding:1rem;border-radius:8px;margin-top:1.5rem;}div.card-body{display:flex;gap:1.5rem;flex-wrap:wrap;}div.card-body ul{margin:0;padding-left:1.2rem;}img{max-width:360px;border:1px solid #ddd;padding:4px;border-radius:4px;display:block;margin:auto;}</style>",
        "</head><body>",
        "<h1>oncotarget-lite Scorecard</h1>",
        "<p>Summary metrics with 95% confidence intervals.</p>",
        _metric_table_html(metrics, bootstrap),
    ]

    for alias in ("GENE1", "GENE2", "GENE3"):
        gene = alias_map.get(alias)
        if not gene:
            continue
        html.append(_describe_gene(gene, alias, ranked, shap_lookup, feature_names))

    html.append("</body></html>")
    ensure_dir(output_path.parent)
    write_text(output_path, "\n".join(html))
    return output_path


def build_docs_index(
    *,
    reports_dir: Path = Path("reports"),
    docs_dir: Path = Path("docs"),
    model_card: Path = Path("oncotarget_lite/model_card.md"),
) -> Path:
    """Create a light HTML index for reviewers and refresh the model card metrics block."""

    metrics = _load_metrics(reports_dir)
    bootstrap = _load_bootstrap(reports_dir)
    scorecard_rel = f"../{SCORECARD_PATH.as_posix()}"
    model_card_rel = f"../{model_card.as_posix()}"
    calibration_plot = f"../{(reports_dir / 'calibration_plot.png').as_posix()}"
    mlflow_ref = _mlflow_link()

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>oncotarget-lite overview</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; max-width: 960px; }}
    table {{ border-collapse: collapse; margin-top: 1rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.5rem 0.8rem; }}
    h1 {{ margin-bottom: 0.5rem; }}
    img {{ max-width: 480px; border: 1px solid #ddd; padding: 4px; border-radius: 4px; margin-top: 1rem; }}
  </style>
</head>
<body>
  <h1>oncotarget-lite Overview</h1>
  <p>Key evaluation metrics with 95% confidence intervals.</p>
  {_metric_table_html(metrics, bootstrap)}
  <p>
    <a href="{scorecard_rel}">Target scorecard</a> ·
    <a href="{model_card_rel}">Responsible AI model card</a>
    {('· <a href="' + mlflow_ref + '">MLflow run</a>' ) if mlflow_ref else ''}
  </p>
  <h2>Calibration</h2>
  <p>Reliability curve from `reports/calibration_plot.png`.</p>
  <img src="{calibration_plot}" alt="Calibration curve" />
</body>
</html>
"""

    ensure_dir(docs_dir)
    output_path = docs_dir / "index.html"
    write_text(output_path, html)
    metrics_table_md = _metric_table_markdown(metrics, bootstrap)
    _update_model_card(model_card, metrics_table_md)
    _update_readme(metrics_table_md)
    return output_path
