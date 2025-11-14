"""FastAPI-based model serving layer with versioning and A/B testing."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
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
from pydantic import ValidationError
import structlog
from prometheus_fastapi_instrumentator import Instrumentator

from ..schemas import APIPredictionRequest, APIPredictionResponse
from ..exceptions import APIError, PredictionError
from ..resilience import get_resilience_manager
from deployment.prediction_service import PredictionService
from ..schemas import APIExplanationResponse

logger = structlog.get_logger()

def setup_logging():
    """Configure structured logging."""
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

resilience_manager = get_resilience_manager()
prediction_service = PredictionService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    setup_logging()
    logger.info("Starting model server...")
    # You can preload models or warm up caches here
    yield
    # Shutdown
    logger.info("Shutting down model server...")


# Create FastAPI app
app = FastAPI(
    title="OncoTarget Lite Model Server",
    description="Production-ready model serving for oncology target prediction",
    version="1.0.0",
    lifespan=lifespan,
)

# Instrument the app with Prometheus metrics
# Add a custom label to distinguish between stable and canary deployments
SERVICE_VERSION = os.environ.get("SERVICE_VERSION", "stable")
Instrumentator().instrument(
    app, metric_namespace="oncotarget_lite", metric_labels={"service_version": SERVICE_VERSION}
).expose(app)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=APIPredictionResponse)
async def predict(request: APIPredictionRequest):
    """Make a single prediction with resilience and validation."""
    try:
        return prediction_service.predict_single(request)
    except PredictionError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.post("/explain", response_model=APIExplanationResponse)
async def explain(request: APIPredictionRequest):
    """Generate a real-time explanation for a single prediction."""
    try:
        return prediction_service.explain_single(request)
    except PredictionError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during explanation: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during explanation")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return prediction_service.health_check()
