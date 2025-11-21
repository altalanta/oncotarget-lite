import streamlit as st
import pandas as pd
from datetime import datetime
from dataclasses import asdict

from oncotarget_lite.feedback import FeedbackStore, FeedbackItem
from deployment.prediction_service import PredictionService

st.set_page_config(page_title="User Feedback", layout="wide")
st.title("🧑‍🔬 User Feedback & Human-in-the-Loop")

st.markdown("""
This page allows expert users to provide feedback on model predictions. This feedback is
invaluable for tracking the real-world performance of the model and for creating a high-quality
dataset for future retraining.
""")

# Initialize services
feedback_store = FeedbackStore()
# Note: In a real production system, this would make a live API call to the model server.
# For this self-contained dashboard, we'll instantiate the service directly.
try:
    prediction_service = PredictionService()
    st.session_state.prediction_service_loaded = True
except Exception as e:
    st.error(f"Failed to load prediction service: {e}")
    st.session_state.prediction_service_loaded = False


# --- Prediction and Feedback Form ---
if st.session_state.prediction_service_loaded:
    st.subheader("Get a Prediction and Provide Feedback")

    gene_input = st.text_input("Enter a Gene Symbol to get a prediction:", "EGFR")

    if st.button("Get Prediction"):
        try:
            # This is a simplification. The prediction service expects a full feature vector.
            # In a real app, you would have a feature retrieval step here.
            # For now, we'll simulate a prediction.
            simulated_features = {"feature1": 0.5} # Placeholder
            request = {"features": simulated_features}
            
            # A real call would look like this:
            # result = prediction_service.predict_single(APIPredictionRequest(**request))
            # For now, we simulate the result to avoid needing a full feature pipeline.
            import random
            prediction = random.random()
            st.session_state.current_prediction = prediction
            st.session_state.current_gene = gene_input

        except Exception as e:
            st.error(f"Failed to get prediction: {e}")

    if 'current_prediction' in st.session_state:
        st.write(f"**Prediction for `{st.session_state.current_gene}`:** `{st.session_state.current_prediction:.4f}`")

        with st.form("feedback_form"):
            feedback = st.selectbox("Your assessment:", ["Helpful", "Not Helpful", "Correct", "Incorrect"])
            comment = st.text_area("Additional comments (optional):")
            submitted = st.form_submit_button("Submit Feedback")

            if submitted:
                feedback_item = FeedbackItem(
                    gene=st.session_state.current_gene,
                    prediction=st.session_state.current_prediction,
                    feedback=feedback,
                    comment=comment,
                    timestamp=datetime.now()
                )
                feedback_store.add_feedback(feedback_item)
                st.success("Thank you for your feedback!")
                # Clear the state
                del st.session_state.current_prediction
                del st.session_state.current_gene

# --- Display Recent Feedback ---
st.subheader("Recent Feedback")
all_feedback = feedback_store.get_all_feedback()

if not all_feedback:
    st.info("No feedback has been submitted yet.")
else:
    df = pd.DataFrame([asdict(item) for item in all_feedback])
    st.dataframe(df)
