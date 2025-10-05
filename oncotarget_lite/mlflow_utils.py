"""MLflow utilities for run resolution and artifact downloading."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import _mlflow, ensure_dir


def resolve_run(selector: str, tracking_uri: str = "mlruns") -> str:
    """Resolve run selector to MLflow run ID.
    
    Args:
        selector: Can be a tag (best=True), run ID, or git_sha=<sha>
        tracking_uri: MLflow tracking URI
    
    Returns:
        MLflow run ID
    """
    mlflow = _mlflow()
    mlflow.set_tracking_uri(str(Path.cwd() / tracking_uri))
    
    # If selector looks like a run ID, return as-is
    if len(selector) == 32 and all(c in '0123456789abcdef' for c in selector):
        return selector
    
    # Search by tag
    if selector.startswith("best="):
        tag_value = selector.split("=", 1)[1].lower() == "true"
        runs = mlflow.search_runs(
            experiment_ids=["0"],  # Default experiment
            filter_string=f"tags.best = '{tag_value}'",
            order_by=["start_time DESC"],
            max_results=1
        )
        if len(runs) > 0:
            return runs.iloc[0]['run_id']
        else:
            raise ValueError(f"No runs found with tag best={tag_value}")
    
    if selector.startswith("git_sha="):
        git_sha = selector.split("=", 1)[1]
        runs = mlflow.search_runs(
            experiment_ids=["0"],
            filter_string=f"tags.git_commit = '{git_sha}'",
            order_by=["start_time DESC"],
            max_results=1
        )
        if len(runs) > 0:
            return runs.iloc[0]['run_id']
        else:
            raise ValueError(f"No runs found with git_sha={git_sha}")
    
    # If it's a partial run ID, search for matches
    runs = mlflow.search_runs(
        experiment_ids=["0"],
        order_by=["start_time DESC"]
    )
    
    for _, run in runs.iterrows():
        if run['run_id'].startswith(selector):
            return run['run_id']
    
    raise ValueError(f"No run found matching selector: {selector}")


def download_run_artifacts(
    run_id: str,
    cache_dir: Path = Path(".cache/selected_run"),
    tracking_uri: str = "mlruns",
) -> Dict[str, Path]:
    """Download artifacts from an MLflow run to local cache.
    
    Args:
        run_id: MLflow run ID
        cache_dir: Local cache directory
        tracking_uri: MLflow tracking URI
    
    Returns:
        Dictionary mapping artifact names to local paths
    """
    mlflow = _mlflow()
    mlflow.set_tracking_uri(str(Path.cwd() / tracking_uri))
    
    # Clear and create cache directory
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    ensure_dir(cache_dir)
    
    artifacts = {}
    
    try:
        # List all artifacts for the run
        artifact_list = mlflow.list_artifacts(run_id)
        
        for artifact in artifact_list:
            artifact_path = artifact.path
            local_path = cache_dir / artifact_path
            
            # Download artifact
            mlflow.download_artifacts(
                run_id=run_id,
                path=artifact_path,
                dst_path=str(cache_dir)
            )
            
            artifacts[artifact_path] = local_path
        
        # Download run metadata
        run = mlflow.get_run(run_id)
        metadata = {
            "run_id": run_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "tags": dict(run.data.tags),
        }
        
        metadata_path = cache_dir / "run_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        artifacts["run_metadata.json"] = metadata_path
        
    except Exception as e:
        raise RuntimeError(f"Failed to download artifacts for run {run_id}: {e}")
    
    return artifacts


def get_run_metadata(run_id: str, tracking_uri: str = "mlruns") -> Dict[str, Any]:
    """Get metadata for an MLflow run.
    
    Args:
        run_id: MLflow run ID
        tracking_uri: MLflow tracking URI
    
    Returns:
        Run metadata dictionary
    """
    mlflow = _mlflow()
    mlflow.set_tracking_uri(str(Path.cwd() / tracking_uri))
    
    run = mlflow.get_run(run_id)
    return {
        "run_id": run_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": dict(run.data.tags),
    }


def list_recent_runs(
    limit: int = 10,
    tracking_uri: str = "mlruns",
    experiment_name: str = "oncotarget-lite",
) -> list[Dict[str, Any]]:
    """List recent MLflow runs.
    
    Args:
        limit: Maximum number of runs to return
        tracking_uri: MLflow tracking URI
        experiment_name: Experiment name
    
    Returns:
        List of run metadata dictionaries
    """
    mlflow = _mlflow()
    mlflow.set_tracking_uri(str(Path.cwd() / tracking_uri))
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            return []
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=limit
        )
        
        return [
            {
                "run_id": row['run_id'],
                "status": row['status'],
                "start_time": row['start_time'],
                "metrics": {k: v for k, v in row.items() if k.startswith('metrics.')},
                "params": {k: v for k, v in row.items() if k.startswith('params.')},
                "tags": {k: v for k, v in row.items() if k.startswith('tags.')},
            }
            for _, row in runs.iterrows()
        ]
    
    except Exception:
        return []


def materialize_artifacts_for_app(
    run_selector: str = "best=True",
    cache_dir: Path = Path(".cache/selected_run"),
) -> Dict[str, Any]:
    """Materialize artifacts for Streamlit app.
    
    Args:
        run_selector: Run selector (tag, ID, or git sha)
        cache_dir: Cache directory
    
    Returns:
        Dictionary with materialized artifact paths and metadata
    """
    try:
        run_id = resolve_run(run_selector)
        artifacts = download_run_artifacts(run_id, cache_dir)
        metadata = get_run_metadata(run_id)
        
        return {
            "run_id": run_id,
            "artifacts": artifacts,
            "metadata": metadata,
            "cache_dir": cache_dir,
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "run_id": None,
            "artifacts": {},
            "metadata": {},
            "cache_dir": cache_dir,
        }