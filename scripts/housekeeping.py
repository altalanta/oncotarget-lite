"""Housekeeping utilities for runtime monitoring and optimization."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, List
import subprocess
import sys


def check_runtime_performance() -> Dict[str, Any]:
    """Check pipeline runtime performance and suggest optimizations."""
    performance_data = {
        "total_runtime": None,
        "stage_runtimes": {},
        "bottlenecks": [],
        "optimizations": [],
        "within_target": False
    }
    
    # Check if we have MLflow data to estimate runtime
    try:
        import mlflow
        tracking_uri = Path.cwd() / "mlruns"
        mlflow.set_tracking_uri(str(tracking_uri))
        
        runs = mlflow.search_runs(
            experiment_ids=["0"],
            order_by=["start_time DESC"],
            max_results=5
        )
        
        if len(runs) > 0:
            latest_run = runs.iloc[0]
            start_time = latest_run.get('start_time')
            end_time = latest_run.get('end_time')
            
            if start_time and end_time:
                runtime_ms = end_time - start_time
                runtime_minutes = runtime_ms / (1000 * 60)
                performance_data["total_runtime"] = runtime_minutes
                performance_data["within_target"] = runtime_minutes <= 10.0
                
                if runtime_minutes > 10.0:
                    performance_data["bottlenecks"].append(
                        f"Pipeline runtime {runtime_minutes:.1f} min exceeds 10 min target"
                    )
                    performance_data["optimizations"].extend([
                        "Consider reducing n_bootstrap in ablations",
                        "Reduce background_size in SHAP explanations",
                        "Downsample synthetic data for development",
                        "Cache intermediate results with DVC"
                    ])
    
    except Exception as e:
        performance_data["bottlenecks"].append(f"Could not check MLflow runtime: {e}")
    
    return performance_data


def check_data_sizes() -> Dict[str, Any]:
    """Check data file sizes and suggest optimizations."""
    size_data = {
        "data_dir_size_mb": 0,
        "large_files": [],
        "recommendations": []
    }
    
    data_dir = Path("data")
    if data_dir.exists():
        total_size = 0
        for file_path in data_dir.rglob("*"):
            if file_path.is_file():
                file_size = file_path.stat().st_size
                total_size += file_size
                
                # Flag files larger than 50MB
                if file_size > 50 * 1024 * 1024:
                    size_data["large_files"].append({
                        "path": str(file_path),
                        "size_mb": file_size / (1024 * 1024)
                    })
        
        size_data["data_dir_size_mb"] = total_size / (1024 * 1024)
        
        if size_data["data_dir_size_mb"] > 500:
            size_data["recommendations"].append(
                "Large data directory - consider using DVC remote storage"
            )
        
        if size_data["large_files"]:
            size_data["recommendations"].append(
                "Large files detected - consider compression or sampling for development"
            )
    
    return size_data


def check_dependencies() -> Dict[str, Any]:
    """Check for missing optional dependencies and suggest installations."""
    deps_data = {
        "missing_optional": [],
        "version_warnings": [],
        "recommendations": []
    }
    
    # Check for XGBoost
    try:
        import xgboost
        version = xgboost.__version__
        print(f"✅ XGBoost {version} available")
    except ImportError:
        deps_data["missing_optional"].append("xgboost")
        deps_data["recommendations"].append(
            "Install XGBoost for better ablation performance: pip install xgboost"
        )
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 11):
        deps_data["version_warnings"].append(
            f"Python {python_version.major}.{python_version.minor} - recommend 3.11+ for best performance"
        )
    
    return deps_data


def optimize_ablation_configs() -> List[str]:
    """Suggest optimizations for ablation configs to meet runtime targets."""
    optimizations = []
    
    configs_dir = Path("configs/ablations")
    if configs_dir.exists():
        for config_file in configs_dir.glob("*.yaml"):
            try:
                import yaml
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                
                # Check for expensive settings
                if config.get("model", {}).get("type") == "mlp":
                    params = config.get("model", {}).get("params", {})
                    max_iter = params.get("max_iter", 200)
                    if max_iter > 200:
                        optimizations.append(
                            f"{config_file.name}: Reduce MLP max_iter from {max_iter} to 200"
                        )
                
                if config.get("model", {}).get("type") == "xgb":
                    params = config.get("model", {}).get("params", {})
                    n_estimators = params.get("n_estimators", 100)
                    if n_estimators > 100:
                        optimizations.append(
                            f"{config_file.name}: Reduce XGB n_estimators from {n_estimators} to 100"
                        )
                
                eval_config = config.get("evaluation", {})
                n_bootstrap = eval_config.get("n_bootstrap", 1000)
                if n_bootstrap > 1000:
                    optimizations.append(
                        f"{config_file.name}: Reduce n_bootstrap from {n_bootstrap} to 1000"
                    )
            
            except Exception as e:
                optimizations.append(f"Could not analyze {config_file.name}: {e}")
    
    return optimizations


def generate_housekeeping_report() -> Dict[str, Any]:
    """Generate comprehensive housekeeping report."""
    report = {
        "timestamp": time.time(),
        "performance": check_runtime_performance(),
        "data_sizes": check_data_sizes(),
        "dependencies": check_dependencies(),
        "config_optimizations": optimize_ablation_configs(),
        "summary": {
            "status": "unknown",
            "critical_issues": [],
            "recommendations": []
        }
    }
    
    # Determine overall status
    critical_issues = []
    all_recommendations = []
    
    # Check performance
    if not report["performance"]["within_target"]:
        if report["performance"]["total_runtime"]:
            critical_issues.append(
                f"Runtime {report['performance']['total_runtime']:.1f} min exceeds 10 min target"
            )
    
    # Collect all recommendations
    all_recommendations.extend(report["performance"]["optimizations"])
    all_recommendations.extend(report["data_sizes"]["recommendations"])
    all_recommendations.extend(report["dependencies"]["recommendations"])
    all_recommendations.extend(report["config_optimizations"])
    
    # Set status
    if critical_issues:
        report["summary"]["status"] = "needs_attention"
    elif all_recommendations:
        report["summary"]["status"] = "optimizable"
    else:
        report["summary"]["status"] = "healthy"
    
    report["summary"]["critical_issues"] = critical_issues
    report["summary"]["recommendations"] = all_recommendations
    
    return report


def main():
    """Main housekeeping function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run housekeeping checks")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    report = generate_housekeeping_report()
    
    if args.json:
        print(json.dumps(report, indent=2))
        return
    
    if not args.quiet:
        print("🧹 Housekeeping Report")
        print("=" * 50)
        
        # Performance
        perf = report["performance"]
        if perf["total_runtime"]:
            status = "✅" if perf["within_target"] else "⚠️"
            print(f"{status} Runtime: {perf['total_runtime']:.1f} min (target: ≤10 min)")
        
        # Data sizes
        sizes = report["data_sizes"]
        print(f"📊 Data directory size: {sizes['data_dir_size_mb']:.1f} MB")
        if sizes["large_files"]:
            print(f"📁 Large files: {len(sizes['large_files'])}")
        
        # Dependencies
        deps = report["dependencies"]
        if deps["missing_optional"]:
            print(f"📦 Missing optional deps: {', '.join(deps['missing_optional'])}")
        
        # Summary
        summary = report["summary"]
        status_emoji = {"healthy": "✅", "optimizable": "⚠️", "needs_attention": "❌"}
        print(f"\n{status_emoji.get(summary['status'], '❓')} Overall status: {summary['status']}")
        
        if summary["critical_issues"]:
            print("\n❌ Critical Issues:")
            for issue in summary["critical_issues"]:
                print(f"  - {issue}")
        
        if summary["recommendations"]:
            print(f"\n💡 Recommendations ({len(summary['recommendations'])}):")
            for rec in summary["recommendations"][:5]:  # Show top 5
                print(f"  - {rec}")
            if len(summary["recommendations"]) > 5:
                print(f"  ... and {len(summary['recommendations']) - 5} more")


if __name__ == "__main__":
    main()