import streamlit as st
import pandas as pd
from oncotarget_lite.experimentation import ExperimentManager

st.set_page_config(page_title="Experiment History", layout="wide")
st.title("🔬 Experiment History")

st.markdown("""
This page provides a summary of all past ML experiments. You can use this to review the results of different
hyperparameter tuning runs and track the evolution of your models.
""")

try:
    exp_manager = ExperimentManager()
    experiments = exp_manager.list_experiments()

    if not experiments:
        st.info("No experiments found. Run an experiment to see the results here.")
    else:
        st.subheader("Experiment Summary")
        df = pd.DataFrame(experiments)
        
        # Display a formatted table
        st.dataframe(df.style.format({
            "best_score": "{:.4f}",
            "created_at": lambda t: pd.to_datetime(t).strftime('%Y-%m-%d %H:%M:%S')
        }))

        # Allow selecting an experiment for more details
        exp_ids = [exp['experiment_id'] for exp in experiments]
        selected_exp_id = st.selectbox("Select an experiment to view details:", exp_ids)

        if selected_exp_id:
            st.subheader(f"Details for Experiment: `{selected_exp_id}`")
            exp = exp_manager.get_experiment(selected_exp_id)
            if exp and exp.best_trial:
                st.write("#### Best Trial")
                st.json(exp.best_trial.parameters)
                st.dataframe(pd.DataFrame([exp.best_trial.metrics]))
            elif exp:
                st.write("This experiment has no completed trials yet.")
            else:
                st.error("Could not load experiment details.")

except Exception as e:
    st.error(f"Could not load experiment history. Have you run any experiments yet? Error: {e}")

