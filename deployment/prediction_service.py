"""Prediction service that encapsulates model loading and inference logic."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import pickle
import shap

from ..exceptions import PredictionError
from ..schemas import APIPredictionRequest
from .model_loader import ModelLoader

logger = logging.getLogger(__name__)


class PredictionService:
    """Encapsulates the model and logic for making predictions."""

    def __init__(self):
        self.model_loader = ModelLoader()
        self.model = None
        self.explainer = None
        self.model_version: str = "unknown"
        self.last_loaded: datetime | None = None
        self._load_model_and_explainer()

    def _load_model_and_explainer(self):
        """Load the production model and explainer from the registry or local files."""
        try:
            self.model = self.model_loader.load_model(stage="Production")
            # In a real scenario, the explainer would also be versioned and loaded from a registry.
            # For now, we'll load it from a fixed path.
            explainer_path = Path("reports/shap/explainer.pkl")
            if explainer_path.exists():
                with open(explainer_path, "rb") as f:
                    self.explainer = pickle.load(f)
            else:
                logger.warning("SHAP explainer not found. Explainability will be disabled.")
            
            # This is a simplification. In a real system, you'd get the version from the model registry.
            self.model_version = "1.0.0" 
            self.last_loaded = datetime.now()
            logger.info("Successfully loaded model and explainer.")
        except Exception as e:
            logger.error(f"Failed to load model or explainer: {e}")
            raise

    def predict_single(self, request: APIPredictionRequest) -> Dict[str, Any]:
        """Make a single prediction."""
        if self.model is None:
            raise PredictionError("Model is not loaded")

        try:
            features_df = pd.DataFrame([request.features])
            prediction = self.model.predict_proba(features_df)[0, 1]
            return {"prediction": float(prediction), "model_version": self.model_version}
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise PredictionError("Failed to make a prediction")

    def explain_single(self, request: APIPredictionRequest) -> Dict[str, Any]:
        """Generate a single explanation."""
        if self.explainer is None:
            raise PredictionError("Explainer is not loaded")

        try:
            features_df = pd.DataFrame([request.features])
            shap_values = self.explainer(features_df)
            
            # For a single prediction, shap_values.values will have shape (1, num_features)
            contributions = pd.Series(shap_values.values[0], index=features_df.columns)
            
            # Return top 10 contributing features
            top_contributions = contributions.abs().nlargest(10)
            top_features = contributions.loc[top_contributions.index].to_dict()

            return {"model_version": self.model_version, "feature_contributions": top_features}
        except Exception as e:
            logger.error(f"Explanation failed: {e}")
            raise PredictionError("Failed to generate explanation")

    def health_check(self) -> Dict[str, Any]:
        """Return the health status of the service."""
        return {
            "status": "ok",
            "model_status": "loaded" if self.model is not None else "not_loaded",
            "explainer_status": "loaded" if self.explainer is not None else "not_loaded",
            "last_updated": self.last_loaded.isoformat() if self.last_loaded else "N/A",
        }




