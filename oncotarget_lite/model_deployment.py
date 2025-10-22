"""Model deployment utilities for versioning and production deployment."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import typer

from .automated_retraining import RetrainResult
from .model import MODELS_DIR
from .utils import ensure_dir

try:
    from .model_server import ModelVersion, ModelRegistry
except ImportError:
    # Fallback if server components not available
    ModelVersion = None
    ModelRegistry = None


def create_model_version(
    run_id: str,
    model_path: Path,
    performance_metrics: Dict[str, float],
    model_type: str,
    feature_names: List[str],
    is_production: bool = False
) -> str:
    """Create a versioned model deployment."""

    # Create version directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_id = f"model_{timestamp}_{run_id[:8]}"
    version_dir = MODELS_DIR / version_id

    ensure_dir(version_dir)

    try:
        # Copy model files
        if model_path.exists():
            if model_path.is_file():
                # Single file model
                dest_path = version_dir / model_path.name
                shutil.copy2(model_path, dest_path)
            else:
                # Directory model
                for file_path in model_path.iterdir():
                    if file_path.is_file():
                        dest_path = version_dir / file_path.name
                        shutil.copy2(file_path, dest_path)

        # Create model metadata
        metadata = {
            "version_id": version_id,
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "performance_metrics": performance_metrics,
            "model_type": model_type,
            "feature_names": feature_names,
            "is_production": is_production,
            "is_active": True,
            "deployment_type": "automated" if run_id.startswith("automated_retrain") else "manual"
        }

        metadata_file = version_dir / "model_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        typer.echo(f"✅ Created model version: {version_id}")
        typer.echo(f"   Path: {version_dir}")
        typer.echo(f"   Performance: {performance_metrics}")

        return version_id

    except Exception as e:
        typer.echo(f"❌ Failed to create model version: {e}")
        # Clean up on failure
        if version_dir.exists():
            shutil.rmtree(version_dir)
        raise


def deploy_to_production(version_id: str, confirm: bool = True) -> bool:
    """Deploy a model version to production."""

    version_dir = MODELS_DIR / version_id
    if not version_dir.exists():
        typer.echo(f"❌ Model version {version_id} not found")
        return False

    # Load metadata
    metadata_file = version_dir / "model_metadata.json"
    if not metadata_file.exists():
        typer.echo(f"❌ Model metadata not found for {version_id}")
        return False

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    if confirm:
        typer.echo(f"🚀 Deploying model {version_id} to production:")
        typer.echo(f"   Model type: {metadata['model_type']}")
        typer.echo(f"   Performance: {metadata['performance_metrics']}")
        typer.echo(f"   Created: {metadata['created_at']}")

        if not typer.confirm("Continue with deployment?"):
            typer.echo("Deployment cancelled")
            return False

    try:
        # Update production symlink/directory
        production_dir = MODELS_DIR / "production"

        # Backup current production if it exists
        if production_dir.exists():
            backup_dir = MODELS_DIR / f"production_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(production_dir, backup_dir)
            typer.echo(f"✅ Backed up current production to: {backup_dir}")

        # Create new production deployment
        if production_dir.exists():
            shutil.rmtree(production_dir)
        shutil.copytree(version_dir, production_dir)

        # Update metadata to mark as production
        metadata["is_production"] = True
        metadata["production_deployed_at"] = datetime.now().isoformat()

        with open(production_dir / "model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        typer.echo(f"✅ Successfully deployed {version_id} to production")
        typer.echo(f"   Production path: {production_dir}")

        return True

    except Exception as e:
        typer.echo(f"❌ Deployment failed: {e}")
        return False


def rollback_production(target_version_id: str, confirm: bool = True) -> bool:
    """Rollback production model to a previous version."""

    production_dir = MODELS_DIR / "production"
    target_dir = MODELS_DIR / target_version_id

    if not target_dir.exists():
        typer.echo(f"❌ Target version {target_version_id} not found")
        return False

    if not production_dir.exists():
        typer.echo("❌ No current production model to rollback")
        return False

    if confirm:
        # Show current and target info
        current_metadata = json.loads((production_dir / "model_metadata.json").read_text())
        target_metadata = json.loads((target_dir / "model_metadata.json").read_text())

        typer.echo("🔄 Rolling back production model:")
        typer.echo(f"   Current: {current_metadata['version_id']} ({current_metadata['created_at']})")
        typer.echo(f"   Target:  {target_metadata['version_id']} ({target_metadata['created_at']})")

        if not typer.confirm("Continue with rollback?"):
            typer.echo("Rollback cancelled")
            return False

    try:
        # Backup current production
        backup_dir = MODELS_DIR / f"production_backup_rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(production_dir, backup_dir)

        # Replace production with target version
        shutil.rmtree(production_dir)
        shutil.copytree(target_dir, production_dir)

        # Update metadata
        metadata = json.loads((production_dir / "model_metadata.json").read_text())
        metadata["is_production"] = True
        metadata["production_deployed_at"] = datetime.now().isoformat()
        metadata["rolled_back_from"] = current_metadata["version_id"]

        with open(production_dir / "model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        typer.echo(f"✅ Successfully rolled back to {target_version_id}")
        typer.echo(f"   Backup saved to: {backup_dir}")

        return True

    except Exception as e:
        typer.echo(f"❌ Rollback failed: {e}")
        return False


def list_model_versions(show_details: bool = False) -> None:
    """List all available model versions."""

    if not MODELS_DIR.exists():
        typer.echo("❌ Models directory not found")
        return

    versions = []
    for version_dir in MODELS_DIR.iterdir():
        if version_dir.is_dir() and not version_dir.name.startswith('.'):
            metadata_file = version_dir / "model_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    versions.append({
                        "version_id": metadata["version_id"],
                        "created_at": metadata["created_at"],
                        "model_type": metadata["model_type"],
                        "is_production": metadata["is_production"],
                        "is_active": metadata.get("is_active", True),
                        "performance": metadata.get("performance_metrics", {})
                    })

                except Exception as e:
                    typer.echo(f"⚠️  Error reading metadata for {version_dir.name}: {e}")

    if not versions:
        typer.echo("No model versions found")
        return

    # Sort by creation date (newest first)
    versions.sort(key=lambda x: x["created_at"], reverse=True)

    typer.echo(f"📋 Model Versions ({len(versions)} total)")
    typer.echo("=" * 60)

    for version in versions:
        status = "🏭 PROD" if version["is_production"] else "✅ ACTIVE" if version["is_active"] else "❌ INACTIVE"

        typer.echo(f"{status} {version['version_id']}")
        typer.echo(f"   Created: {version['created_at']}")
        typer.echo(f"   Type: {version['model_type']}")

        if show_details and version["performance"]:
            typer.echo("   Performance:")
            for metric, value in version["performance"].items():
                typer.echo(f"     {metric.upper()}: {value:.3f}")

        typer.echo()


def integrate_with_retraining(retrain_result: RetrainResult) -> str:
    """Integrate model deployment with automated retraining results."""

    if not retrain_result.success:
        typer.echo(f"❌ Cannot deploy failed retraining result: {retrain_result.error_message}")
        return ""

    # Create model version from retraining result
    version_id = create_model_version(
        run_id=retrain_result.new_model_version,
        model_path=Path("models") / "logreg_pipeline.pkl",  # Default path assumption
        performance_metrics={
            "auroc": retrain_result.new_performance.get("auroc", 0),
            "ap": retrain_result.new_performance.get("ap", 0),
            "accuracy": retrain_result.new_performance.get("accuracy", 0),
            "f1": retrain_result.new_performance.get("f1", 0)
        },
        model_type="logreg",  # Default assumption
        feature_names=[],  # Would need to extract from actual model
        is_production=retrain_result.deployed
    )

    if retrain_result.deployed:
        typer.echo(f"✅ Model {version_id} deployed to production via automated retraining")

    return version_id


def cleanup_old_versions(keep_production: bool = True, keep_recent: int = 5, dry_run: bool = False) -> int:
    """Clean up old model versions, keeping production and recent versions."""

    if not MODELS_DIR.exists():
        typer.echo("❌ Models directory not found")
        return 0

    versions = []
    for version_dir in MODELS_DIR.iterdir():
        if version_dir.is_dir() and not version_dir.name.startswith('.'):
            metadata_file = version_dir / "model_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    versions.append({
                        "dir": version_dir,
                        "metadata": metadata,
                        "is_production": metadata.get("is_production", False),
                        "created_at": metadata.get("created_at", "")
                    })

                except Exception as e:
                    typer.echo(f"⚠️  Error reading metadata for {version_dir.name}: {e}")

    if not versions:
        typer.echo("No model versions found")
        return 0

    # Sort by creation date (oldest first for cleanup)
    versions.sort(key=lambda x: x["created_at"])

    to_delete = []
    kept_count = 0

    for version in versions:
        keep = False

        if keep_production and version["is_production"]:
            keep = True
            kept_count += 1
        elif kept_count < keep_recent:
            keep = True
            kept_count += 1

        if not keep:
            to_delete.append(version)

    if not to_delete:
        typer.echo("No old versions to clean up")
        return 0

    if dry_run:
        typer.echo(f"🔍 Would delete {len(to_delete)} old model versions:")
        for version in to_delete:
            typer.echo(f"   - {version['metadata']['version_id']}")
        return len(to_delete)

    # Actually delete
    deleted_count = 0
    for version in to_delete:
        try:
            if version["dir"].exists():
                shutil.rmtree(version["dir"])
                typer.echo(f"🗑️  Deleted: {version['metadata']['version_id']}")
                deleted_count += 1
        except Exception as e:
            typer.echo(f"⚠️  Failed to delete {version['dir'].name}: {e}")

    typer.echo(f"✅ Cleaned up {deleted_count} old model versions")
    return deleted_count


# CLI Commands
def deploy_cmd(
    version_id: str = typer.Argument(..., help="Model version ID to deploy"),
    confirm: bool = typer.Option(True, help="Confirm deployment"),
) -> None:
    """Deploy a model version to production."""
    success = deploy_to_production(version_id, confirm=confirm)
    if not success:
        raise typer.Exit(1)


def rollback_cmd(
    target_version: str = typer.Argument(..., help="Target version ID to rollback to"),
    confirm: bool = typer.Option(True, help="Confirm rollback"),
) -> None:
    """Rollback production model to a previous version."""
    success = rollback_production(target_version, confirm=confirm)
    if not success:
        raise typer.Exit(1)


def versions_cmd(
    details: bool = typer.Option(False, help="Show detailed performance metrics"),
) -> None:
    """List all available model versions."""
    list_model_versions(show_details=details)


def cleanup_cmd(
    keep_production: bool = typer.Option(True, help="Keep production models"),
    keep_recent: int = typer.Option(5, help="Keep N most recent models"),
    dry_run: bool = typer.Option(False, help="Show what would be deleted"),
) -> None:
    """Clean up old model versions."""
    deleted_count = cleanup_old_versions(
        keep_production=keep_production,
        keep_recent=keep_recent,
        dry_run=dry_run
    )

    if not dry_run:
        typer.echo(f"Deleted {deleted_count} old model versions")
    else:
        typer.echo(f"Would delete {deleted_count} old model versions")


def server_cmd(
    host: str = typer.Option("0.0.0.0", help="Server host"),
    port: int = typer.Option(8000, help="Server port"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
) -> None:
    """Start the model serving server."""
    try:
        from .model_server import run_server
        typer.echo(f"🚀 Starting model server on {host}:{port}")
        run_server(host=host, port=port, reload=reload)
    except ImportError as e:
        typer.echo(f"❌ Server dependencies not available: {e}")
        typer.echo("Install with: pip install fastapi uvicorn")
        raise typer.Exit(1)
