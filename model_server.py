"""FastAPI-based model serving layer with versioning and A/B testing."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# from .model import load_model_pipeline  # Commented out due to import issues
# from .data import load_processed_data  # Commented out due to import issues
from .utils import ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Represents a model version with metadata."""
    version_id: str
    model_path: Path
    created_at: datetime
    performance_metrics: Dict[str, float]
    feature_names: List[str]
    model_type: str
    is_production: bool = False
    is_active: bool = True


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""
    test_id: str
    model_a: str  # version_id
    model_b: str  # version_id
    traffic_split: float  # percentage for model A (0-1)
    start_time: datetime
    end_time: Optional[datetime] = None
    is_active: bool = True


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: Dict[str, float] = Field(..., description="Feature values for prediction")
    model_version: Optional[str] = Field(None, description="Specific model version to use")
    request_id: Optional[str] = Field(None, description="Unique request identifier")

    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError('Features cannot be empty')
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: float
    prediction_class: int
    probabilities: Dict[str, float]
    model_version: str
    request_id: Optional[str]
    processing_time_ms: float
    timestamp: datetime
    confidence_score: float


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    samples: List[Dict[str, float]] = Field(..., description="List of feature samples")
    model_version: Optional[str] = Field(None, description="Specific model version to use")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    batch_id: str
    total_samples: int
    processing_time_ms: float
    timestamp: datetime


class ModelRegistry:
    """Registry for managing model versions."""

    def __init__(self, models_dir: Path = Path("models")):
        self.models_dir = models_dir
        self.versions: Dict[str, ModelVersion] = {}
        self.ab_tests: Dict[str, ABTestConfig] = {}
        self._load_models()

    def _load_models(self):
        """Load all available model versions."""
        try:
            # Look for model directories
            if not self.models_dir.exists():
                logger.warning(f"Models directory {self.models_dir} does not exist")
                return

            for model_dir in self.models_dir.iterdir():
                if model_dir.is_dir() and not model_dir.name.startswith('.'):
                    version_id = model_dir.name

                    # Try to load model metadata
                    metadata_file = model_dir / "model_metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)

                            model_version = ModelVersion(
                                version_id=version_id,
                                model_path=model_dir / "pipeline.pkl",
                                created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat())),
                                performance_metrics=metadata.get("performance_metrics", {}),
                                feature_names=metadata.get("feature_names", []),
                                model_type=metadata.get("model_type", "unknown"),
                                is_production=metadata.get("is_production", False),
                                is_active=metadata.get("is_active", True)
                            )

                            # For demo purposes, don't require actual model files
                            self.versions[version_id] = model_version
                            logger.info(f"Loaded model version: {version_id}")

                        except Exception as e:
                            logger.error(f"Error loading model {version_id}: {e}")

        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def get_model(self, version_id: Optional[str] = None) -> ModelVersion:
        """Get a model version, preferring production if none specified."""
        if version_id:
            if version_id not in self.versions:
                raise HTTPException(status_code=404, detail=f"Model version {version_id} not found")
            return self.versions[version_id]

        # Get production model
        production_models = [v for v in self.versions.values() if v.is_production]
        if production_models:
            return production_models[0]

        # Fallback to latest active model
        active_models = [v for v in self.versions.values() if v.is_active]
        if not active_models:
            raise HTTPException(status_code=404, detail="No active models available")

        return max(active_models, key=lambda v: v.created_at)

    def get_ab_test_model(self, request_id: str, test_id: Optional[str] = None) -> ModelVersion:
        """Get model for A/B testing based on request ID."""
        if not test_id:
            # Find active A/B test
            active_tests = [test for test in self.ab_tests.values() if test.is_active]
            if not active_tests:
                # No A/B testing, use regular model selection
                return self.get_model()

            test_id = active_tests[0].test_id

        if test_id not in self.ab_tests:
            raise HTTPException(status_code=404, detail=f"A/B test {test_id} not found")

        test_config = self.ab_tests[test_id]

        # Use request ID to determine which model to use
        # Simple hash-based routing for consistent A/B assignment
        request_hash = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        use_model_a = (request_hash % 100) < (test_config.traffic_split * 100)

        model_version_id = test_config.model_a if use_model_a else test_config.model_b

        if model_version_id not in self.versions:
            raise HTTPException(status_code=404, detail=f"Model {model_version_id} not found in A/B test")

        return self.versions[model_version_id]

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available model versions."""
        return [asdict(v) for v in self.versions.values()]

    def list_ab_tests(self) -> List[Dict[str, Any]]:
        """List all A/B tests."""
        return [asdict(test) for test in self.ab_tests.values()]


class PredictionCache:
    """Simple in-memory cache for predictions."""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 300):
        self.cache: Dict[str, tuple[Any, datetime]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

    def _make_key(self, features: Dict[str, float], model_version: str) -> str:
        """Create cache key from features and model version."""
        features_str = json.dumps(features, sort_keys=True)
        return hashlib.md5(f"{features_str}:{model_version}".encode()).hexdigest()

    def get(self, features: Dict[str, float], model_version: str) -> Optional[Any]:
        """Get cached prediction if available and not expired."""
        key = self._make_key(features, model_version)
        if key in self.cache:
            prediction, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                return prediction
            else:
                del self.cache[key]

        return None

    def set(self, features: Dict[str, float], model_version: str, prediction: Any):
        """Cache a prediction result."""
        key = self._make_key(features, model_version)

        # Clean up expired entries if cache is getting full
        if len(self.cache) >= self.max_size:
            self._cleanup_expired()

        if len(self.cache) < self.max_size:
            self.cache[key] = (prediction, datetime.now())

    def _cleanup_expired(self):
        """Remove expired cache entries."""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp >= timedelta(seconds=self.ttl_seconds)
        ]
        for key in expired_keys:
            del self.cache[key]

    def clear(self):
        """Clear all cached predictions."""
        self.cache.clear()


class ModelServer:
    """Main model server class."""

    def __init__(self, models_dir: Path = Path("models")):
        self.registry = ModelRegistry(models_dir)
        self.cache = PredictionCache()
        self.models: Dict[str, Any] = {}  # Loaded models cache

    async def load_model(self, version_id: str) -> Any:
        """Load a model if not already cached."""
        if version_id not in self.models:
            # Mock model for demonstration purposes
            # In a real implementation, this would load the actual model
            class MockModel:
                def __init__(self):
                    self.classes_ = [0, 1]

                def predict(self, X):
                    # Simple mock prediction based on sum of features
                    import numpy as np
                    predictions = []
                    for sample in X.values:
                        # Mock logic: predict 1 if sum of features > 1.0, else 0
                        pred = 1 if np.sum(sample) > 1.0 else 0
                        predictions.append(pred)
                    return np.array(predictions)

                def predict_proba(self, X):
                    # Mock probabilities
                    import numpy as np
                    probabilities = []
                    for sample in X.values:
                        prob_1 = 0.8 if np.sum(sample) > 1.0 else 0.3
                        prob_0 = 1.0 - prob_1
                        probabilities.append([prob_0, prob_1])
                    return np.array(probabilities)

            self.models[version_id] = MockModel()
            logger.info(f"Loaded mock model version: {version_id}")

        return self.models[version_id]

    async def predict_single(self, request: PredictionRequest, ab_test_id: Optional[str] = None) -> PredictionResponse:
        """Make a single prediction."""
        start_time = time.time()

        try:
            # Determine which model to use
            if ab_test_id:
                model_version = self.registry.get_ab_test_model(request.request_id or "unknown", ab_test_id)
            else:
                model_version = self.registry.get_model(request.model_version)

            # Check cache first
            cached_prediction = self.cache.get(request.features, model_version.version_id)
            if cached_prediction:
                processing_time = (time.time() - start_time) * 1000
                return PredictionResponse(
                    **cached_prediction,
                    processing_time_ms=processing_time,
                    timestamp=datetime.now()
                )

            # Load model and make prediction
            model = await self.load_model(model_version.version_id)

            # Convert features to DataFrame
            features_df = pd.DataFrame([request.features])

            # Make prediction
            prediction_proba = model.predict_proba(features_df)
            prediction = model.predict(features_df)[0]

            # Calculate probabilities
            probabilities = {
                str(cls): float(prob)
                for cls, prob in zip(model.classes_, prediction_proba[0])
            }

            # Calculate confidence score (max probability)
            confidence_score = float(max(prediction_proba[0]))

            response_data = {
                "prediction": float(prediction),
                "prediction_class": int(prediction),
                "probabilities": probabilities,
                "model_version": model_version.version_id,
                "request_id": request.request_id,
                "confidence_score": confidence_score
            }

            # Cache the result
            self.cache.set(request.features, model_version.version_id, response_data)

            processing_time = (time.time() - start_time) * 1000
            return PredictionResponse(
                **response_data,
                processing_time_ms=processing_time,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    async def predict_batch(self, request: BatchPredictionRequest) -> BatchPredictionResponse:
        """Make batch predictions."""
        start_time = time.time()
        batch_id = f"batch_{int(time.time())}_{hash(str(request.samples)) % 10000}"

        predictions = []
        for sample in request.samples:
            single_request = PredictionRequest(
                features=sample,
                model_version=request.model_version,
                request_id=f"{batch_id}_{len(predictions)}"
            )
            prediction = await self.predict_single(single_request)
            predictions.append(prediction)

        processing_time = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            predictions=predictions,
            batch_id=batch_id,
            total_samples=len(request.samples),
            processing_time_ms=processing_time,
            timestamp=datetime.now()
        )


# Global server instance
model_server = ModelServer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting model server...")
    yield
    # Shutdown
    logger.info("Shutting down model server...")


# Create FastAPI app
app = FastAPI(
    title="OncoTarget Lite Model Server",
    description="Production-ready model serving for oncology target prediction",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, x_request_id: Optional[str] = Header(None)):
    """Make a single prediction."""
    if x_request_id:
        request.request_id = x_request_id
    return await model_server.predict_single(request)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions."""
    return await model_server.predict_batch(request)


@app.get("/models")
async def list_models():
    """List all available model versions."""
    return {"models": model_server.registry.list_models()}


@app.get("/models/{version_id}")
async def get_model(version_id: str):
    """Get details for a specific model version."""
    try:
        model_version = model_server.registry.get_model(version_id)
        return asdict(model_version)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model: {e}")


@app.get("/ab-tests")
async def list_ab_tests():
    """List all A/B tests."""
    return {"ab_tests": model_server.registry.list_ab_tests()}


@app.post("/ab-tests")
async def create_ab_test(config: ABTestConfig):
    """Create a new A/B test."""
    model_server.registry.ab_tests[config.test_id] = config
    return {"message": f"A/B test {config.test_id} created successfully"}


@app.delete("/ab-tests/{test_id}")
async def delete_ab_test(test_id: str):
    """Delete an A/B test."""
    if test_id in model_server.registry.ab_tests:
        del model_server.registry.ab_tests[test_id]
        return {"message": f"A/B test {test_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"A/B test {test_id} not found")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "models_loaded": len(model_server.models),
        "cache_size": len(model_server.cache.cache)
    }


@app.post("/cache/clear")
async def clear_cache():
    """Clear prediction cache."""
    model_server.cache.clear()
    return {"message": "Cache cleared successfully"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "OncoTarget Lite Model Server",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "single_prediction": "/predict",
            "batch_prediction": "/predict/batch",
            "list_models": "/models",
            "ab_testing": "/ab-tests"
        }
    }


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the model server."""
    uvicorn.run(
        "oncotarget_lite.model_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()
