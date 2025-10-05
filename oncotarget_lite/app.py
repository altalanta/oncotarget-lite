"""Streamlit triage UI for oncotarget-lite model results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from .mlflow_utils import materialize_artifacts_for_app, list_recent_runs, resolve_run


def check_auth() -> bool:
    """Simple token-based authentication."""
    token = os.getenv("ONCOTARGET_APP_TOKEN")
    if not token:
        st.warning("⚠️ ONCOTARGET_APP_TOKEN not set. Running in open mode for CI screenshot.")
        return True
    
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        with st.form("auth_form"):
            entered_token = st.text_input("Enter access token:", type="password")
            submitted = st.form_submit_button("Authenticate")
            
            if submitted:
                if entered_token == token:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid token")
                    return False
        return False
    
    return True


@st.cache_data
def load_metrics_data() -> Dict[str, Any]:
    """Load metrics from reports directory."""
    try:
        metrics_file = Path("reports/metrics.json")
        if metrics_file.exists():
            with open(metrics_file) as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Failed to load metrics: {e}")
    return {}


@st.cache_data
def load_ablations_data() -> Optional[pd.DataFrame]:
    """Load ablations metrics CSV."""
    try:
        ablations_file = Path("reports/ablations/metrics.csv")
        if ablations_file.exists():
            return pd.read_csv(ablations_file)
    except Exception as e:
        st.error(f"Failed to load ablations data: {e}")
    return None


@st.cache_data
def load_shap_data() -> Dict[str, Any]:
    """Load SHAP data and images."""
    shap_data = {"global_summary": None, "examples": {}, "feature_names": []}
    
    try:
        # Load global summary image
        global_summary = Path("reports/shap/global_summary.png")
        if global_summary.exists():
            shap_data["global_summary"] = str(global_summary)
        
        # Load example images
        shap_dir = Path("reports/shap")
        if shap_dir.exists():
            for img_file in shap_dir.glob("example_*.png"):
                gene_name = img_file.stem.replace("example_", "")
                shap_data["examples"][gene_name] = str(img_file)
        
        # Load feature names from alias map
        alias_map_file = Path("reports/shap/alias_map.json")
        if alias_map_file.exists():
            with open(alias_map_file) as f:
                alias_map = json.load(f)
                shap_data["feature_names"] = list(alias_map.keys())
    
    except Exception as e:
        st.error(f"Failed to load SHAP data: {e}")
    
    return shap_data


@st.cache_data
def load_predictions_data() -> Optional[pd.DataFrame]:
    """Load predictions data."""
    try:
        predictions_file = Path("reports/predictions.parquet")
        if predictions_file.exists():
            return pd.read_parquet(predictions_file)
    except Exception as e:
        st.error(f"Failed to load predictions: {e}")
    return None


def filter_targets_by_pathway(predictions_df: pd.DataFrame, pathway_filter: str) -> pd.DataFrame:
    """Filter targets by pathway or GO term."""
    if not pathway_filter:
        return predictions_df
    
    # Simple string matching - in a real app, you'd use proper GO/pathway annotations
    return predictions_df[
        predictions_df['gene_id'].str.contains(pathway_filter, case=False, na=False) |
        predictions_df.get('pathway', pd.Series(dtype=str)).str.contains(pathway_filter, case=False, na=False)
    ]


def render_ranking_view(predictions_df: pd.DataFrame, min_score: float, uncertainty_threshold: float):
    """Render the target ranking view."""
    st.subheader("🎯 Target Ranking")
    
    # Filter predictions
    filtered_df = predictions_df[
        (predictions_df['predicted_prob'] >= min_score) &
        (predictions_df.get('uncertainty', 0) <= uncertainty_threshold)
    ].copy()
    
    # Sort by predicted probability
    filtered_df = filtered_df.sort_values('predicted_prob', ascending=False)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Targets", len(predictions_df))
    with col2:
        st.metric("Filtered Targets", len(filtered_df))
    with col3:
        if len(predictions_df) > 0:
            st.metric("Pass Rate", f"{len(filtered_df)/len(predictions_df):.1%}")
    
    # Display top targets
    if len(filtered_df) > 0:
        st.dataframe(
            filtered_df[['gene_id', 'predicted_prob', 'actual_label']].head(20),
            use_container_width=True
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download filtered targets (CSV)",
            data=csv,
            file_name="filtered_targets.csv",
            mime="text/csv"
        )
    else:
        st.info("No targets match the current filters.")


def render_shap_view(shap_data: Dict[str, Any], selected_gene: str):
    """Render SHAP explanations view."""
    st.subheader("🔍 Model Explanations (SHAP)")
    
    if shap_data["global_summary"]:
        st.subheader("Global Feature Importance")
        st.image(shap_data["global_summary"], caption="Global SHAP feature importance")
    
    if selected_gene and selected_gene in shap_data["examples"]:
        st.subheader(f"Example: {selected_gene}")
        st.image(shap_data["examples"][selected_gene], caption=f"SHAP explanation for {selected_gene}")
    elif shap_data["examples"]:
        st.subheader("Example Explanations")
        example_gene = list(shap_data["examples"].keys())[0]
        st.image(shap_data["examples"][example_gene], caption=f"SHAP explanation for {example_gene}")


def render_comparison_view(ablations_df: Optional[pd.DataFrame]):
    """Render model comparison view."""
    st.subheader("⚖️ Model Comparison")
    
    if ablations_df is not None:
        # Model performance comparison
        st.subheader("Model Performance")
        chart_data = ablations_df[['experiment', 'test_auroc', 'test_ap']].set_index('experiment')
        st.bar_chart(chart_data)
        
        # Feature impact comparison
        st.subheader("Feature Impact")
        feature_types = ablations_df['feature_type'].value_counts()
        st.bar_chart(feature_types)
        
        # Detailed table
        st.subheader("Detailed Results")
        display_cols = ['experiment', 'model_type', 'feature_type', 'feature_count', 'test_auroc', 'test_ap']
        st.dataframe(ablations_df[display_cols], use_container_width=True)
        
        # Link to ablations summary
        summary_file = Path("reports/ablations/summary.html")
        if summary_file.exists():
            st.markdown("📊 [View detailed ablation analysis](reports/ablations/summary.html)")
    else:
        st.info("No ablation results available. Run ablations with `make ablations`.")


def render_mlflow_selector() -> str:
    """Render MLflow run selector."""
    st.sidebar.subheader("🎛️ Model Run Selection")
    
    # Get recent runs
    recent_runs = list_recent_runs(limit=5)
    
    if recent_runs:
        run_options = []
        for run in recent_runs:
            run_id_short = run["run_id"][:8]
            auroc = run["metrics"].get("metrics.auroc", "N/A")
            timestamp = run["start_time"]
            run_options.append(f"{run_id_short} (AUROC: {auroc}) - {timestamp}")
        
        selected_idx = st.sidebar.selectbox(
            "Select MLflow run:",
            range(len(run_options)),
            format_func=lambda i: run_options[i]
        )
        
        return recent_runs[selected_idx]["run_id"]
    else:
        # Manual run ID entry
        return st.sidebar.text_input(
            "Enter MLflow run ID:",
            value="best=True",
            help="Use 'best=True', run ID, or 'git_sha=<sha>'"
        )


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="OncoTarget Lite - Triage UI",
        page_icon="🎯",
        layout="wide"
    )
    
    # Authentication
    if not check_auth():
        return
    
    st.title("🎯 OncoTarget Lite - Target Triage Interface")
    st.markdown("Explore model predictions, explanations, and ablation results")
    
    # Sidebar controls
    st.sidebar.title("🎛️ Controls")
    
    # Model run selection
    selected_run = render_mlflow_selector()
    
    # Filter controls
    st.sidebar.subheader("🔍 Filters")
    pathway_filter = st.sidebar.text_input("Pathway/GO term filter:", "")
    min_score = st.sidebar.slider("Minimum prediction score:", 0.0, 1.0, 0.5, 0.05)
    uncertainty_threshold = st.sidebar.slider("Max uncertainty:", 0.0, 1.0, 1.0, 0.05)
    
    # View selection
    view_mode = st.sidebar.radio(
        "View Mode:",
        ["🎯 Ranking", "🔍 Per-target SHAP", "⚖️ Compare Runs"]
    )
    
    # Load data
    with st.spinner("Loading data..."):
        metrics_data = load_metrics_data()
        predictions_df = load_predictions_data()
        shap_data = load_shap_data()
        ablations_df = load_ablations_data()
    
    # Display current model metrics
    if metrics_data:
        st.subheader("📊 Current Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("AUROC", f"{metrics_data.get('auroc', 0):.3f}")
        with col2:
            st.metric("Average Precision", f"{metrics_data.get('ap', 0):.3f}")
        with col3:
            st.metric("Brier Score", f"{metrics_data.get('brier', 0):.3f}")
        with col4:
            st.metric("Calibration ECE", f"{metrics_data.get('ece', 0):.3f}")
    
    # Main content area
    if predictions_df is not None:
        # Apply pathway filter
        if pathway_filter:
            predictions_df = filter_targets_by_pathway(predictions_df, pathway_filter)
        
        # Render selected view
        if view_mode == "🎯 Ranking":
            render_ranking_view(predictions_df, min_score, uncertainty_threshold)
        
        elif view_mode == "🔍 Per-target SHAP":
            selected_gene = st.selectbox(
                "Select gene for SHAP analysis:",
                options=predictions_df['gene_id'].unique()[:50],  # Limit for performance
                index=0
            )
            render_shap_view(shap_data, selected_gene)
        
        elif view_mode == "⚖️ Compare Runs":
            render_comparison_view(ablations_df)
    
    else:
        st.error("No predictions data available. Run the pipeline first with `make all`.")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("📋 **Links**")
    
    # Link to target scorecard
    scorecard_file = Path("reports/target_scorecard.html")
    if scorecard_file.exists():
        st.sidebar.markdown("📊 [Target Scorecard](reports/target_scorecard.html)")
    
    # Link to model card
    model_card_file = Path("oncotarget_lite/model_card.md")
    if model_card_file.exists():
        st.sidebar.markdown("📄 [Model Card](oncotarget_lite/model_card.md)")
    
    st.sidebar.markdown("🔬 *Generated by oncotarget-lite*")


if __name__ == "__main__":
    main()