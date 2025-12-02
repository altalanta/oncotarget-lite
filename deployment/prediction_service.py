"""Prediction service that encapsulates model loading and inference logic.

This service is designed for dependency injection:
- Accepts configuration through constructor parameters
- Supports lazy model loading
- Can be easily mocked for testing
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from oncotarget_lite.exceptions import PredictionError
from oncotarget_lite.schemas import APIPredictionRequest
from oncotarget_lite.logging_config import get_logger, log_timing

logger = get_logger(__name__)


class PredictionService:
    """Encapsulates the model and logic for making predictions.

    This class supports dependency injection through constructor parameters,
    making it easy to configure and test.

    Args:
        models_dir: Directory containing model files
        explainer_path: Path to SHAP explainer pickle file
        model_name: Name of the model in the registry
        model_stage: Stage to load from registry (e.g., "Production")
        lazy_load: If True, defer model loading until first prediction

    Example:
        # Production usage
        service = PredictionService()

        # Testing with custom paths
        service = PredictionService(
            models_dir=Path("test_models"),
            lazy_load=False,
        )
    """

    def __init__(
        self,
        models_dir: Path = Path("models"),
        explainer_path: Optional[Path] = None,
        model_name: str = "oncotarget-lite",
        model_stage: str = "Production",
        lazy_load: bool = False,
    ):
        self._models_dir = models_dir
        self._explainer_path = explainer_path or Path("reports/shap/explainer.pkl")
        self._model_name = model_name
        self._model_stage = model_stage

        self._model = None
        self._explainer = None
        self._model_loader = None
        self._model_version: str = "unknown"
        self._last_loaded: Optional[datetime] = None
        self._is_loaded = False

        if not lazy_load:
            self._ensure_loaded()

    def _get_model_loader(self):
        """Get or create the model loader."""
        if self._model_loader is None:
            from deployment.model_loader import ModelLoader
            self._model_loader = ModelLoader(models_dir=self._models_dir)
        return self._model_loader

    def _ensure_loaded(self) -> None:
        """Ensure the model and explainer are loaded."""
        if self._is_loaded:
            return

        try:
            logger.info("loading_model_started")
            loader = self._get_model_loader()
            self._model = loader.load_model(
                model_name=self._model_name,
                stage=self._model_stage,
            )

            # Load explainer if available
            if self._explainer_path.exists():
                with open(self._explainer_path, "rb") as f:
                    self._explainer = pickle.load(f)
                logger.info("explainer_loaded", path=str(self._explainer_path))
            else:
                logger.warning(
                    "explainer_not_found",
                    path=str(self._explainer_path),
                    message="Explainability will be disabled",
                )

            # Get version from model metadata if available
            self._model_version = "1.0.0"
            self._last_loaded = datetime.now()
            self._is_loaded = True

            logger.info(
                "model_loaded_successfully",
                model_version=self._model_version,
                has_explainer=self._explainer is not None,
            )
        except Exception as e:
            logger.error(
                "model_loading_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

    @property
    def model(self):
        """Get the loaded model, loading it if necessary."""
        self._ensure_loaded()
        return self._model

    @property
    def explainer(self):
        """Get the loaded explainer, loading it if necessary."""
        self._ensure_loaded()
        return self._explainer

    @property
    def model_version(self) -> str:
        """Get the model version."""
        return self._model_version

    @property
    def last_loaded(self) -> Optional[datetime]:
        """Get the timestamp of when the model was last loaded."""
        return self._last_loaded

    @property
    def is_loaded(self) -> bool:
        """Check if the model has been loaded."""
        return self._is_loaded

    @log_timing("model_inference")
    def predict_single(self, request: APIPredictionRequest) -> Dict[str, Any]:
        """Make a single prediction.

        Args:
            request: The prediction request containing features

        Returns:
            Dictionary with prediction and model version

        Raises:
            PredictionError: If prediction fails
        """
        self._ensure_loaded()

        if self._model is None:
            raise PredictionError("Model is not loaded")

        try:
            features_df = pd.DataFrame([request.features])
            prediction = self._model.predict_proba(features_df)[0, 1]
            return {"prediction": float(prediction), "model_version": self._model_version}
        except Exception as e:
            logger.error(
                "inference_failed",
                error_type=type(e).__name__,
                error_message=str(e),
                feature_count=len(request.features),
            )
            raise PredictionError("Failed to make a prediction")

    @log_timing("shap_explanation")
    def explain_single(self, request: APIPredictionRequest) -> Dict[str, Any]:
        """Generate a single explanation.

        Args:
            request: The prediction request containing features

        Returns:
            Dictionary with model version and feature contributions

        Raises:
            PredictionError: If explanation generation fails
        """
        self._ensure_loaded()

        if self._explainer is None:
            raise PredictionError("Explainer is not loaded")

        try:
            features_df = pd.DataFrame([request.features])
            shap_values = self._explainer(features_df)

            # For a single prediction, shap_values.values will have shape (1, num_features)
            contributions = pd.Series(shap_values.values[0], index=features_df.columns)

            # Return top 10 contributing features
            top_contributions = contributions.abs().nlargest(10)
            top_features = contributions.loc[top_contributions.index].to_dict()

            return {"model_version": self._model_version, "feature_contributions": top_features}
        except Exception as e:
            logger.error(
                "explanation_failed",
                error_type=type(e).__name__,
                error_message=str(e),
                feature_count=len(request.features),
            )
            raise PredictionError("Failed to generate explanation")

    def health_check(self) -> Dict[str, Any]:
        """Return the health status of the service.

        Returns:
            Dictionary with status information
        """
        return {
            "status": "ok",
            "model_status": "loaded" if self._is_loaded and self._model is not None else "not_loaded",
            "explainer_status": "loaded" if self._is_loaded and self._explainer is not None else "not_loaded",
            "last_updated": self._last_loaded.isoformat() if self._last_loaded else "N/A",
        }

    def reload(self) -> None:
        """Force reload of the model and explainer."""
        self._is_loaded = False
        self._model = None
        self._explainer = None
        self._ensure_loaded()
