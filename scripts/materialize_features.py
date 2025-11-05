#!/usr/bin/env python3
"""
Script to materialize (pre-calculate and store) the full set of engineered features.
"""
import sys
from pathlib import Path
import pandas as pd

# Add the project root to the Python path to allow for module imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncotarget_lite.features.orchestrator import FeatureOrchestrator
from oncotarget_lite.utils import ensure_dir

def main():
    """
    Runs the feature materialization process.
    """
    print("🚀 Starting feature materialization...")

    # Define paths
    processed_data_dir = Path("data/processed")
    output_path = processed_data_dir / "engineered_features.parquet"
    ensure_dir(processed_data_dir)

    # Load the raw data (e.g., the gene list from the training data)
    try:
        labels_df = pd.read_parquet(processed_data_dir / "labels.parquet")
        genes = labels_df['gene']
        print(f"Loaded {len(genes)} genes from the training data.")
    except FileNotFoundError:
        print("Error: Could not find processed labels file at 'data/processed/labels.parquet'.")
        print("Please run the data preparation pipeline first.")
        sys.exit(1)

    # Initialize the feature orchestrator
    orchestrator = FeatureOrchestrator()

    # Extract all features
    print("🔬 Extracting all engineered features. This may take a while...")
    features_df = orchestrator.extract_all_features(genes, cache_key="full_dataset")

    # Save the materialized features
    features_df.to_parquet(output_path)
    print(f"✅ Successfully materialized {len(features_df.columns)} features for {len(features_df)} genes.")
    print(f"   Saved to: {output_path}")

    # Print a summary
    summary = orchestrator.get_feature_summary(features_df)
    print("\n📊 Feature Summary:")
    for category, count in summary["feature_categories"].items():
        print(f"  - {category.capitalize()}: {count} features")

if __name__ == "__main__":
    main()




