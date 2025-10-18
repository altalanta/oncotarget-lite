#!/usr/bin/env python3
"""
Demo script for testing the enhanced interpretability dashboard functionality.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from oncotarget_lite.interpretability_dashboard import InterpretabilityDashboard


def test_interpretability_dashboard():
    """Test the enhanced interpretability dashboard functionality."""

    print("🎨 Testing Enhanced Interpretability Dashboard for oncotarget-lite")
    print("=" * 65)

    # Test dashboard creation
    shap_dir = Path("reports/shap")

    if not shap_dir.exists():
        print("❌ SHAP directory not found. Please run 'make explain' first.")
        return

    print(f"📊 Loading SHAP data from {shap_dir}...")

    try:
        dashboard = InterpretabilityDashboard(shap_dir)

        if dashboard.genes is None or len(dashboard.genes) == 0:
            print("❌ No SHAP data found. Please run 'make explain' first.")
            return

        print(f"✅ Loaded SHAP data: {len(dashboard.genes)} genes, {len(dashboard.feature_names)} features")

        # Test individual plot creation
        print("\n🖼️  Testing individual visualizations...")

        # Global importance plot
        global_fig = dashboard.create_global_importance_plot()
        print(f"   ✅ Global importance plot: {len(global_fig.data)} traces")

        # Gene contribution plot (for first gene)
        if dashboard.genes is not None and len(dashboard.genes) > 0:
            gene_fig = dashboard.create_gene_contribution_plot(dashboard.genes[0])
            print(f"   ✅ Gene contribution plot: {len(gene_fig.data)} traces")

        # Feature interaction heatmap
        interaction_fig = dashboard.create_feature_interaction_heatmap()
        print(f"   ✅ Feature interaction heatmap: {len(interaction_fig.data)} traces")

        # Create comprehensive dashboard
        print("\n📋 Creating comprehensive dashboard...")
        dashboard_path = Path("reports/dashboard/interpretability_dashboard.html")
        dashboard.create_comprehensive_dashboard(dashboard_path)
        print(f"   ✅ Dashboard created: {dashboard_path}")

        # Test static exports
        print("\n🖼️  Creating static exports...")
        static_dir = Path("reports/dashboard/static")
        dashboard.save_static_exports(static_dir)
        print(f"   ✅ Static exports created: {static_dir}")

        # Test model comparison (if validation reports exist)
        validation_reports = list(Path("reports").glob("validation_report_*.json"))
        if validation_reports:
            print(f"\n🔍 Testing model comparison ({len(validation_reports)} reports found)...")
            comparison_path = Path("reports/dashboard/model_comparison.html")
            dashboard.create_model_comparison_dashboard(validation_reports, comparison_path)
            print(f"   ✅ Model comparison dashboard: {comparison_path}")
        else:
            print("\n🔍 Skipping model comparison (no validation reports found)")

        print("\n🎉 Enhanced interpretability dashboard demo completed!")
        print("Open the generated HTML files in your browser to explore the interactive visualizations.")

    except Exception as e:
        print(f"❌ Error testing dashboard: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_interpretability_dashboard()
