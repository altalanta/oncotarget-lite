"""Streamlit application for exploring synthetic immunotherapy target scores."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from oncotarget.eval import (
    classification_summary,
    compute_reliability_curve,
    compute_shap_values,
    summarize_bootstrap,
)
from oncotarget.features import FeatureMatrix, create_feature_matrix
from oncotarget.io import merge_gene_feature_table
from oncotarget.model import MLPConfig, train_mlp
from oncotarget.scoring import explain_gene_score, score_candidates
from oncotarget.viz import (
    create_calibration_plot,
    create_essentiality_violin,
    create_shap_importance,
    create_tumor_vs_normal_bar,
)

st.set_page_config(page_title="oncotarget-lite", layout="wide")


@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, FeatureMatrix, pd.DataFrame]:
    merged = merge_gene_feature_table()
    feature_matrix = create_feature_matrix(merged)
    scores = score_candidates(feature_matrix.features, feature_matrix.labels)
    return merged, feature_matrix, scores


@st.cache_resource(show_spinner=False)
def fit_model(features: pd.DataFrame, labels: pd.Series):
    metrics, model = train_mlp(features, labels, MLPConfig(epochs=120, patience=15))
    return metrics, model


merged, feature_matrix, scores = load_data()
metrics, model = fit_model(feature_matrix.features, feature_matrix.labels)

st.title("oncotarget-lite dashboard")
st.caption("Synthetic end-to-end immunotherapy target explorer – for demonstration only.")

gene_options = list(merged.index)
tumor_options = [col.replace("tumor_", "") for col in merged.columns if col.startswith("tumor_")]

st.sidebar.header("Controls")
selected_genes = st.sidebar.multiselect("Gene symbols", gene_options, default=scores.head(3).index.tolist())
selected_tumor = st.sidebar.selectbox("Tumor indication", tumor_options, index=0)
show_shap = st.sidebar.checkbox("Compute SHAP explanations", value=False)

st.sidebar.info(
    "Scores combine tumor vs normal expression, dependency screens, and annotations. "
    "Use the ADC heuristic to sanity-check antibody conjugate suitability."
)

st.subheader("Scorecard overview")
st.dataframe(scores[["score", "rank", "log2fc_component", "low_normal_component"]].head(10))

# Model predictions for calibration
with st.spinner("Generating model diagnostics..."):
    with np.errstate(divide="ignore", invalid="ignore"):
        import torch

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(feature_matrix.features.values, dtype=torch.float32))
            probs = torch.sigmoid(logits).numpy().squeeze()
    y_true = feature_matrix.labels.astype(int).to_numpy()
    summary = classification_summary(y_true, probs)
    bootstrap = summarize_bootstrap(y_true, probs)
    calibration_df = compute_reliability_curve(y_true, probs)

col1, col2, col3 = st.columns(3)
col1.metric("AUROC", f"{summary['auroc']:.3f}")
col2.metric("AUPRC", f"{summary['auprc']:.3f}")
col3.metric("Brier", f"{summary['brier']:.3f}")

ci_text = ", ".join(
    f"{name.upper()} 95% CI: {vals.lower:.2f} – {vals.upper:.2f}"
    for name, vals in bootstrap.items()
)
st.caption(ci_text)

calibration_fig = create_calibration_plot(calibration_df)
st.pyplot(calibration_fig)
plt.close(calibration_fig)

def adc_heuristic(row: pd.Series) -> tuple[str, str]:
    surface = bool(row.get("is_cell_surface", False))
    signal = bool(row.get("signal_peptide", False))
    ig = bool(row.get("ig_like_domain", False))
    length = float(row.get("protein_length", np.nan))
    if surface and signal and ig and 200 <= length <= 800:
        return "green", "Meets key ADC-friendly hallmarks"
    if surface and signal and 120 <= length <= 1500:
        return "yellow", "Surface-accessible but requires domain review"
    return "red", "Unfavorable surface biology for ADC"

if not selected_genes:
    selected_genes = scores.head(3).index.tolist()

for gene in selected_genes:
    st.markdown(f"### {gene}")
    breakdown = explain_gene_score(gene, scores)
    heur_color, heur_text = adc_heuristic(merged.loc[gene])
    st.write(pd.DataFrame({"component": list(breakdown.keys()), "value": list(breakdown.values())}))
    st.markdown(
        f"**ADC heuristic:** :{heur_color}[{heur_text}]"
    )

    bar_fig = create_tumor_vs_normal_bar(gene, merged)
    st.pyplot(bar_fig)
    plt.close(bar_fig)

    violin_fig = create_essentiality_violin(gene, merged)
    st.plotly_chart(violin_fig, use_container_width=True)

    tumor_col = f"tumor_{selected_tumor.upper()}"
    normal_cols = [col for col in merged.columns if col.startswith("normal_")]
    st.markdown(
        f"Tumor median ({selected_tumor}): {merged.loc[gene, tumor_col]:.2f} TPM · "
        f"Min normal: {merged.loc[gene, normal_cols].min():.2f} TPM"
    )

if show_shap:
    with st.spinner("Running SHAP kernel explainer (slow for larger datasets)..."):
        shap_values = compute_shap_values(model, feature_matrix.features)
    if shap_values is not None:
        shap_fig = create_shap_importance(shap_values)
        st.plotly_chart(shap_fig, use_container_width=True)
    else:
        st.warning("SHAP not available in this environment.")

st.markdown(
    "---\n"
    "This dashboard ships with cached synthetic data. Replace CSVs in `data/raw/` "
    "with harmonized cohorts to drive bespoke target discovery experiments."
)
