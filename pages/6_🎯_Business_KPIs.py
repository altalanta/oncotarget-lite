import streamlit as st
import pandas as pd
import plotly.express as px
from oncotarget_lite.feedback import FeedbackStore

st.set_page_config(page_title="Business KPIs", layout="wide")
st.title("🎯 Business KPI Dashboard")

st.markdown("""
This dashboard provides a high-level view of the model's real-world performance,
based on the collected feedback from expert users. This allows us to track the business
value of the model and its alignment with human expertise.
""")

# Initialize the feedback store
feedback_store = FeedbackStore()
all_feedback = feedback_store.get_all_feedback()

if not all_feedback:
    st.warning("No feedback data available. Please submit feedback via the 'User Feedback' page.")
else:
    df = pd.DataFrame([vars(item) for item in all_feedback])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # --- Main KPI: Expert Agreement Rate ---
    st.subheader("Key Performance Indicator: Expert Agreement Rate")
    
    positive_feedback = ["Correct", "Helpful"]
    df['is_agreement'] = df['feedback'].isin(positive_feedback)
    
    agreement_rate = df['is_agreement'].mean() * 100 if not df.empty else 0
    
    st.metric(
        label="Expert Agreement Rate",
        value=f"{agreement_rate:.2f}%",
        help="Percentage of predictions that expert users marked as 'Correct' or 'Helpful'."
    )
    
    st.progress(int(agreement_rate))

    # --- Visualizations ---
    st.subheader("Feedback Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Feedback over time
        st.write("#### Expert Agreement Over Time")
        df['date'] = df['timestamp'].dt.date
        agreement_over_time = df.groupby('date')['is_agreement'].mean().reset_index()
        agreement_over_time['is_agreement'] *= 100 # Convert to percentage

        fig_time = px.line(
            agreement_over_time,
            x='date',
            y='is_agreement',
            title='Daily Expert Agreement Rate (%)',
            labels={'is_agreement': 'Agreement Rate (%)', 'date': 'Date'}
        )
        st.plotly_chart(fig_time, use_container_width=True)

    with col2:
        # Agreement by prediction score
        st.write("#### Agreement by Prediction Score")
        df['prediction_bin'] = pd.cut(df['prediction'], bins=5)
        agreement_by_bin = df.groupby('prediction_bin')['is_agreement'].mean().reset_index()
        agreement_by_bin['is_agreement'] *= 100 # Convert to percentage
        
        fig_bin = px.bar(
            agreement_by_bin,
            x='prediction_bin',
            y='is_agreement',
            title='Expert Agreement Rate by Model Score',
            labels={'is_agreement': 'Agreement Rate (%)', 'prediction_bin': 'Prediction Score Bin'}
        )
        st.plotly_chart(fig_bin, use_container_width=True)
