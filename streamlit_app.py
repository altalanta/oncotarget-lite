"""
# OncoTarget-Lite: MLOps Dashboard

This dashboard provides a unified view into the health, performance, and governance of the OncoTarget-Lite ML system.
"""
import streamlit as st

st.set_page_config(
    page_title="OncoTarget-Lite MLOps Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 OncoTarget-Lite: MLOps Dashboard")

st.markdown("""
Welcome to the central hub for the OncoTarget-Lite project. This dashboard consolidates all of the key artifacts
from our MLOps pipeline into a single, accessible interface. Use the sidebar to navigate to the different pages.

### Dashboard Pages:
- **📈 Monitoring Report:** View the latest data drift and model performance monitoring report from our production environment.
- **✅ Data Validation:** See the results of our automated data quality checks from the Great Expectations pipeline.
- **📜 Model Card:** Review the official model card for the latest production model, including its performance, characteristics, and intended use.
- **🔬 Experiment History:** Browse the history of all ML experiments and view their results.
""")
