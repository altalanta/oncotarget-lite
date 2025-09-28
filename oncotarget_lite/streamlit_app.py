"""Lightweight Streamlit explainer for oncotarget-lite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from .utils import load_json

REPORTS_DIR = Path("reports")
SHAP_DIR = REPORTS_DIR / "shap"

st.set_page_config(page_title="oncotarget-lite", layout="wide")
st.title("oncotarget-lite quality checks")

metrics_path = REPORTS_DIR / "metrics.json"
if metrics_path.exists():
    metrics = load_json(metrics_path)
    st.subheader("Evaluation metrics")
    st.dataframe(pd.DataFrame([metrics]))
else:
    st.info("Run the pipeline to generate metrics.json before using the app.")

shap_images = sorted(SHAP_DIR.glob("example_*.png"))
if shap_images:
    st.subheader("Example SHAP explanations")
    cols = st.columns(len(shap_images))
    for col, path in zip(cols, shap_images):
        col.image(str(path), caption=path.stem.replace("example_", ""))
else:
    st.info("SHAP images unavailable – run `python -m oncotarget_lite.cli explain`." )
