"""Model loading utilities for production deployment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import mlflow

from oncotarget_lite.utils import ensure_dir
from oncotarget_lite.logging_config import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """Load and manage trained models for inference."""

    def __init__(self, models_dir: Path = Path("models")):
        self.models_dir = models_dir
        ensure_dir(models_dir)
        self.model_cache: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}

    def list_available_models(self) -> list[Dict[str, Any]]:
        """List all available trained models."""
        models = []

        # Check for main model
        main_model_path = self.models_dir / "logreg_pipeline.pkl"
        if main_model_path.exists():
            metadata = self._load_model_metadata("main")
            models.append({
                "name": "main",
                "type": "main",
                "path": str(main_model_path),
                "metadata": metadata,
                "status": "ready"
            })

        # Check for ablation models
        ablations_dir = self.models_dir / "ablations"
        if ablations_dir.exists():
            for ablation_dir in ablations_dir.iterdir():
                if ablation_dir.is_dir():
                    model_path = ablation_dir / "pipeline.pkl"
                    if model_path.exists():
                        metadata = self._load_model_metadata(ablation_dir.name)
                        models.append({
                            "name": ablation_dir.name,
                            "type": "ablation",
                            "path": str(model_path),
                            "metadata": metadata,
                            "status": "ready"
                        })

        logger.debug("models_listed", count=len(models))
        return models

    def _load_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """Load metadata for a specific model."""
        metadata_file = self.models_dir / "ablations" / model_name / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def load_model(self, model_name: str = "oncotarget-lite", stage: str = "Production") -> Any:
        """
        Load a specific model.

        In a production environment, this will load the model from the MLflow Model Registry.
        For local/dev, it can fall back to a local file.
        """
        cache_key = f"{model_name}@{stage}"
        if cache_key in self.model_cache:
            logger.debug("model_cache_hit", model_name=model_name, stage=stage)
            return self.model_cache[cache_key]

        try:
            # Attempt to load from MLflow Model Registry first
            model_uri = f"models:/{model_name}/{stage}"
            logger.info(
                "loading_model_from_registry",
                model_name=model_name,
                stage=stage,
                uri=model_uri,
            )
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(
                "model_loaded_from_registry",
                model_name=model_name,
                stage=stage,
            )
            self.model_cache[cache_key] = model
            return model
        except Exception as e:
            logger.warning(
                "mlflow_registry_load_failed",
                model_name=model_name,
                stage=stage,
                error_type=type(e).__name__,
                error_message=str(e),
            )

            # Fallback to local file for dev/testing
            model_path = self.models_dir / "logreg_pipeline.pkl"
            if not model_path.exists():
                logger.error("local_model_not_found", path=str(model_path))
                raise FileNotFoundError(f"Model not found at local path: {model_path}")

            logger.info("loading_model_from_local", path=str(model_path))
            model = joblib.load(model_path)
            self.model_cache[cache_key] = model
            return model

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model."""
        models = self.list_available_models()
        for model in models:
            if model["name"] == model_name:
                return model

        raise ValueError(f"Model not found: {model_name}")

    def clear_cache(self) -> None:
        """Clear the model cache."""
        self.model_cache.clear()
        logger.info("model_cache_cleared")

    def get_model_performance(self, model_name: str) -> Dict[str, float]:
        """Get performance metrics for a model."""
        metadata = self.get_model_info(model_name)["metadata"]

        if "test_metrics" in metadata:
            return metadata["test_metrics"]

        # Fallback for main model
        if model_name == "main":
            # Try to load from basic metrics file
            metrics_file = Path("reports/metrics_basic.json")
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    return data.get("test", {})

        return {}
