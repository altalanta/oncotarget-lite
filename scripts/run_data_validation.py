#!/usr/bin/env python3
"""
Script to run Great Expectations data validation.
"""
import sys
from great_expectations.data_context import DataContext

def main():
    """
    Runs the Great Expectations checkpoint and exits with a non-zero status code if validation fails.
    """
    print("🚀 Running data validation with Great Expectations...")

    # Initialize the Data Context
    # The base_directory is set to the directory containing great_expectations.yml
    context = DataContext(context_root_dir="great_expectations")

    # Run the checkpoint
    checkpoint_name = "raw_clinical_trials_checkpoint"
    result = context.run_checkpoint(checkpoint_name=checkpoint_name)

    if not result["success"]:
        print("❌ Data validation failed!")
        # Optional: Print a more detailed report or link to Data Docs
        sys.exit(1)
    
    print("✅ Data validation passed!")
    # The validation results and updated Data Docs are saved by the checkpoint's action list.

if __name__ == "__main__":
    main()








