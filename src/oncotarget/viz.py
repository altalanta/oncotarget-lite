"""Visualization helpers for oncogenic target discovery demos."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def create_tumor_vs_normal_bar(gene: str, merged: pd.DataFrame) -> plt.Figure:
    """Bar plot comparing tumor vs. normal expression for a gene."""

    if gene not in merged.index:
        msg = f"gene '{gene}' not present in merged table"
        raise KeyError(msg)

    normal_cols = [col for col in merged.columns if col.startswith("normal_")]
    tumor_cols = [col for col in merged.columns if col.startswith("tumor_")]
    normal_values = merged.loc[gene, normal_cols]
    tumor_values = merged.loc[gene, tumor_cols]

    fig, ax = plt.subplots(figsize=(6, 4))
    x_pos = np.arange(len(normal_cols) + len(tumor_cols))
    values = np.concatenate([normal_values.values, tumor_values.values])
    colors = ["#4C72B0"] * len(normal_cols) + ["#DD8452"] * len(tumor_cols)
    labels = [col.replace("normal_", "Normal ") for col in normal_cols] + [
        col.replace("tumor_", "Tumor ") for col in tumor_cols
    ]
    ax.bar(x_pos, values, color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Median TPM")
    ax.set_title(f"{gene}: tumor vs normal expression")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def create_essentiality_violin(gene: str, merged: pd.DataFrame) -> go.Figure:
    """Plot dependency score distribution for the selected gene."""

    if gene not in merged.index:
        msg = f"gene '{gene}' not present in merged table"
        raise KeyError(msg)
    dep_cols = [col for col in merged.columns if col.startswith("dep_")]
    values = merged.loc[gene, dep_cols].values
    figure = go.Figure()
    figure.add_trace(
        go.Violin(
            y=values,
            box_visible=True,
            meanline_visible=True,
            fillcolor="#8C8C8C",
            line_color="#4C72B0",
            name=gene,
            hovertemplate="Dependency: %{y:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"{gene} dependency scores (lower = more essential)",
        xaxis_title="",
        yaxis_title="CERES score",
        showlegend=False,
        height=350,
    )
    return figure


def create_calibration_plot(calibration_df: pd.DataFrame) -> plt.Figure:
    """Return a calibration curve figure."""

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(
        calibration_df["mean_predicted"],
        calibration_df["fraction_of_positives"],
        marker="o",
        color="#4C72B0",
        label="Model",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Ideal")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def create_shap_importance(shap_values: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart of mean |SHAP| values."""

    importance = shap_values.abs().mean(axis=0).sort_values(ascending=False).head(top_n)
    figure = go.Figure(
        go.Bar(
            x=importance.values[::-1],
            y=importance.index[::-1],
            orientation="h",
            marker_color="#DD8452",
        )
    )
    figure.update_layout(
        title="Mean |SHAP| feature importance",
        xaxis_title="Impact on model output",
        yaxis_title="Feature",
        height=400,
    )
    return figure
