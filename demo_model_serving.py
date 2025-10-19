#!/usr/bin/env python3
"""Demonstration script for model serving and deployment functionality."""

import sys
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from oncotarget_lite.model_deployment import (
    list_model_versions,
    cleanup_old_versions
)

def demo_model_versions():
    """Demonstrate model version management."""
    print("📋 Model Version Management Demo")
    print("=" * 40)

    print("Listing current model versions...")
    list_model_versions(show_details=True)
    print()

def demo_cleanup():
    """Demonstrate model cleanup functionality."""
    print("🧹 Model Cleanup Demo")
    print("=" * 40)

    print("Checking for old model versions to clean up...")
    deleted_count = cleanup_old_versions(
        keep_production=True,
        keep_recent=2,
        dry_run=True
    )

    if deleted_count > 0:
        print(f"Would delete {deleted_count} old model versions")
        print("\nTo actually clean up, run:")
        print("  python -m oncotarget_lite.cli cleanup --keep-recent 2")
    else:
        print("No old versions to clean up")
    print()

def demo_server_info():
    """Demonstrate server information."""
    print("🌐 Model Server Information")
    print("=" * 40)

    print("The model server provides:")
    print("• RESTful API for predictions")
    print("• Model versioning and deployment")
    print("• A/B testing capabilities")
    print("• Caching for improved performance")
    print("• Health monitoring and metrics")
    print()

    print("Server endpoints:")
    print("• POST /predict - Single prediction")
    print("• POST /predict/batch - Batch predictions")
    print("• GET /models - List model versions")
    print("• GET /health - Health check")
    print("• GET /docs - Interactive API documentation")
    print()

def demo_api_usage():
    """Demonstrate API usage examples."""
    print("🔗 API Usage Examples")
    print("=" * 40)

    print("Starting the server:")
    print("  python -m oncotarget_lite.cli serve")
    print("  # Server will be available at http://localhost:8000")
    print()

    print("Example single prediction:")
    print("  curl -X POST 'http://localhost:8000/predict' \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{")
    print("      \"features\": {\"gene_A\": 0.5, \"gene_B\": 0.3, \"clinical_score\": 0.8},")
    print("      \"model_version\": \"model_20250101_123456_abc12345\"")
    print("    }'")
    print()

    print("Example batch prediction:")
    print("  curl -X POST 'http://localhost:8000/predict/batch' \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{")
    print("      \"samples\": [")
    print("        {\"gene_A\": 0.5, \"gene_B\": 0.3, \"clinical_score\": 0.8},")
    print("        {\"gene_A\": 0.2, \"gene_B\": 0.7, \"clinical_score\": 0.4}")
    print("      ]")
    print("    }'")
    print()

def demo_ab_testing():
    """Demonstrate A/B testing setup."""
    print("🔬 A/B Testing Demo")
    print("=" * 40)

    print("A/B testing allows you to compare two model versions:")
    print("1. Create an A/B test configuration")
    print("2. Traffic is automatically split between models")
    print("3. Monitor performance differences")
    print("4. Deploy the winning model")
    print()

    print("Example A/B test creation:")
    print("  curl -X POST 'http://localhost:8000/ab-tests' \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{")
    print("      \"test_id\": \"experiment_001\",")
    print("      \"model_a\": \"model_20250101_123456_abc12345\",")
    print("      \"model_b\": \"model_20250102_154321_def67890\",")
    print("      \"traffic_split\": 0.5,")
    print("      \"start_time\": \"2025-01-01T00:00:00Z\"")
    print("    }'")
    print()

def main():
    """Main demonstration function."""
    print("🚀 Enhanced Model Serving & API Layer Demonstration")
    print("=" * 60)
    print()

    demo_model_versions()
    demo_cleanup()
    demo_server_info()
    demo_api_usage()
    demo_ab_testing()

    print("✅ Model Serving Demonstration completed!")
    print()
    print("Next steps:")
    print("1. Run 'make serve' to start the model server")
    print("2. Visit http://localhost:8000/docs for interactive API docs")
    print("3. Try the prediction endpoints with your data")
    print("4. Use model deployment commands to manage versions")

if __name__ == "__main__":
    main()
