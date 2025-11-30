import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(page_title="Data Validation Report", layout="wide")
st.title("✅ Data Validation Report")

st.markdown("""
This page displays the results from our automated data validation pipeline, powered by Great Expectations.
This "Data Docs" site is automatically generated every time our data validation pipeline runs.
""")

# Path to the Great Expectations Data Docs index page
data_docs_path = Path("great_expectations/uncommitted/data_docs/local_site/index.html")

if data_docs_path.exists():
    with open(data_docs_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    st.subheader("Great Expectations Data Docs")
    # Adjust the base URL for local file paths
    html_content = html_content.replace('src="', 'src="./')
    components.html(html_content, height=800, scrolling=True)
else:
    st.warning("Data Docs not found. Please run the data validation pipeline (`dvc repro`) to generate the report.")






