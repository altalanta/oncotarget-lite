import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Model Card", layout="wide")
st.title("📜 Model Card")

st.markdown("""
This page displays the official model card for the latest version of the OncoTarget-Lite model.
This card is automatically generated and updated with every new model release.
""")

model_card_path = Path("docs/model_card.md")

if model_card_path.exists():
    st.markdown(model_card_path.read_text())
else:
    st.warning("Model card not found. Please run the training and deployment pipeline to generate it.")





