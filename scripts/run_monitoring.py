import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset

def main():
    """
    This script runs a data drift and quality analysis using the Evidently library.
    It compares a reference dataset (assumed to be the training data) with a
    current dataset (assumed to be recent production data) to detect any drift.
    """
    # Load the datasets
    # In a real-world scenario, the reference data would be your training dataset,
    # and the current data would be the data your model is seeing in production.
    try:
        reference_data = pd.read_parquet("data/processed/features.parquet")
        current_data = pd.read_parquet("test_data/features.parquet") # Using test data as a proxy
        
        # Add a dummy target column for demonstration purposes, as drift detection can be run on target values as well
        # In a real scenario, you would join your features with the actual labels/outcomes.
        reference_data['target'] = pd.read_parquet("data/processed/labels.parquet")
        current_data['target'] = pd.read_parquet("test_data/labels.parquet")

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure that 'data/processed/features.parquet', 'test_data/features.parquet', 'data/processed/labels.parquet', and 'test_data/labels.parquet' exist.")
        print("You may need to run your data preparation pipeline first.")
        return

    # Create the Evidently report
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
        DataQualityPreset(),
    ])

    # Run the report
    report.run(reference_data=reference_data, current_data=current_data)

    # Save the report as an HTML file
    report.save_html("reports/monitoring_report.html")
    print("Monitoring report saved to reports/monitoring_report.html")

if __name__ == "__main__":
    main()

