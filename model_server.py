"""FastAPI-based model serving layer with versioning and A/B testing.

This module implements a production-ready async API server for ML inference.
Key design decisions:
- Blocking ML inference calls are offloaded to a thread pool using asyncio.to_thread()
- This keeps the event loop responsive and allows the server to handle concurrent requests
- Health checks remain synchronous as they are lightweight
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import structlog
from prometheus_fastapi_instrumentator import Instrumentator

from oncotarget_lite.schemas import (
    APIPredictionRequest,
    APIPredictionResponse,
    APIExplanationResponse,
)
from oncotarget_lite.exceptions import PredictionError
from oncotarget_lite.resilience import get_resilience_manager
from deployment.prediction_service import PredictionService

logger = structlog.get_logger()


def setup_logging() -> None:
    """Configure structured logging with JSON output for production observability."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


# Initialize global services
# These are created at module load time and reused across requests
resilience_manager = get_resilience_manager()
prediction_service = PredictionService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown tasks:
    - Startup: Configure logging, warm up model caches
    - Shutdown: Clean up resources, flush logs
    """
    # Startup
    setup_logging()
    logger.info("model_server_starting", version=app.version)
    
    # Warm up the model by making a dummy prediction
    # This ensures the model is loaded into memory before the first real request
    try:
        warmup_request = APIPredictionRequest(features={"warmup": 0.0})
        await asyncio.to_thread(prediction_service.predict_single, warmup_request)
        logger.info("model_warmup_complete")
    except Exception as e:
        logger.warning("model_warmup_failed", error=str(e))
    
    yield
    
    # Shutdown
    logger.info("model_server_shutting_down")


# Create FastAPI app
app = FastAPI(
    title="OncoTarget Lite Model Server",
    description="Production-ready model serving for oncology target prediction",
    version="1.0.0",
    lifespan=lifespan,
)

# Instrument the app with Prometheus metrics
SERVICE_VERSION = os.environ.get("SERVICE_VERSION", "stable")
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=APIPredictionResponse)
async def predict(request: APIPredictionRequest) -> Dict[str, Any]:
    """
    Make a single prediction with resilience and validation.
    
    This endpoint offloads the blocking ML inference to a thread pool,
    keeping the async event loop free to handle other concurrent requests.
    
    Args:
        request: The prediction request containing feature values.
        
    Returns:
        A dictionary containing the prediction score and model version.
        
    Raises:
        HTTPException: 500 if prediction fails.
    """
    try:
        # Offload blocking ML inference to thread pool
        # This prevents the prediction from blocking the event loop
        result = await asyncio.to_thread(
            prediction_service.predict_single,
            request
        )
        return result
    except PredictionError as e:
        logger.error("prediction_failed", error=str(e), request_id=id(request))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("prediction_unexpected_error", error=str(e))
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.post("/explain", response_model=APIExplanationResponse)
async def explain(request: APIPredictionRequest) -> Dict[str, Any]:
    """
    Generate a real-time explanation for a single prediction.
    
    This endpoint uses SHAP values to explain which features contributed
    most to the prediction. Like /predict, it offloads the blocking
    computation to a thread pool.
    
    Args:
        request: The prediction request containing feature values.
        
    Returns:
        A dictionary containing the model version and feature contributions.
        
    Raises:
        HTTPException: 500 if explanation generation fails.
    """
    try:
        # Offload blocking SHAP computation to thread pool
        result = await asyncio.to_thread(
            prediction_service.explain_single,
            request
        )
        return result
    except PredictionError as e:
        logger.error("explanation_failed", error=str(e), request_id=id(request))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("explanation_unexpected_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during explanation"
        )


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for Kubernetes liveness/readiness probes.
    
    This endpoint is intentionally kept synchronous and lightweight
    as it's called frequently by the orchestrator. It should return
    quickly even under high load.
    
    Returns:
        A dictionary containing the health status and component states.
    """
    # Health check is lightweight, no need for thread pool
    return prediction_service.health_check()


@app.post("/predict/batch")
async def predict_batch(requests: list[APIPredictionRequest]) -> list[Dict[str, Any]]:
    """
    Make batch predictions for multiple inputs.
    
    This endpoint processes multiple predictions concurrently using
    asyncio.gather(), which can significantly improve throughput
    when handling many requests.
    
    Args:
        requests: A list of prediction requests.
        
    Returns:
        A list of prediction results in the same order as the requests.
        
    Raises:
        HTTPException: 500 if any prediction fails.
    """
    try:
        # Process all predictions concurrently
        # Each prediction is offloaded to the thread pool
        tasks = [
            asyncio.to_thread(prediction_service.predict_single, req)
            for req in requests
        ]
        results = await asyncio.gather(*tasks)
        return list(results)
    except PredictionError as e:
        logger.error("batch_prediction_failed", error=str(e), batch_size=len(requests))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("batch_prediction_unexpected_error", error=str(e))
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
