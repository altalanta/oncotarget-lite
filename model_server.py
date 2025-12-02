"""FastAPI-based model serving layer with dependency injection.

This module implements a production-ready async API server for ML inference.
Key design decisions:
- Dependency injection via FastAPI's Depends() for testability
- Lazy initialization of expensive resources (models)
- Blocking ML inference calls are offloaded to a thread pool
- Centralized structured logging with correlation IDs
- Feature validation against the model's expected schema
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any, Dict

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from oncotarget_lite.schemas import (
    APIPredictionRequest,
    APIPredictionResponse,
    APIExplanationResponse,
    APIFeatureInfoResponse,
    APIValidationErrorResponse,
)
from oncotarget_lite.exceptions import PredictionError
from oncotarget_lite.logging_config import (
    configure_logging,
    get_logger,
    LogContext,
)
from oncotarget_lite.middleware import add_observability_middleware
from oncotarget_lite.feature_validation import (
    FeatureValidator,
    ValidationMode,
)
from oncotarget_lite.dependencies import (
    get_prediction_service,
    get_feature_validator,
    get_container,
)

# Type aliases for dependency injection
from deployment.prediction_service import PredictionService

PredictionServiceDep = Annotated[PredictionService, Depends(get_prediction_service)]
FeatureValidatorDep = Annotated[FeatureValidator, Depends(get_feature_validator)]

# Configure logging at module load (can be reconfigured in lifespan)
configure_logging(environment=os.environ.get("ENVIRONMENT", "development"))
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown tasks:
    - Startup: Configure logging, warm up dependencies
    - Shutdown: Clean up resources, flush logs
    """
    # Startup - reconfigure logging based on environment
    environment = os.environ.get("ENVIRONMENT", "development")
    configure_logging(environment=environment)
    logger.info(
        "model_server_starting",
        version=app.version,
        environment=environment,
    )

    # Get the dependency container and warm up dependencies
    container = get_container()

    # Warm up feature validator (lightweight)
    validator = container.feature_validator
    logger.info(
        "feature_validator_ready",
        feature_count=len(validator.expected_features),
    )

    # Warm up the prediction service (loads model)
    try:
        prediction_service = container.prediction_service
        logger.info(
            "prediction_service_ready",
            model_version=prediction_service.model_version,
            is_loaded=prediction_service.is_loaded,
        )
    except Exception as e:
        logger.warning("prediction_service_warmup_failed", error=str(e))

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

# Add observability middleware (correlation IDs, request logging)
add_observability_middleware(app)

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


def _get_validation_mode(mode_str: str | None) -> ValidationMode:
    """Convert string validation mode to enum."""
    if mode_str is None:
        return ValidationMode.LENIENT
    try:
        return ValidationMode(mode_str)
    except ValueError:
        return ValidationMode.LENIENT


@app.post(
    "/predict",
    response_model=APIPredictionResponse,
    responses={
        422: {"model": APIValidationErrorResponse, "description": "Validation Error"},
    },
)
async def predict(
    request: APIPredictionRequest,
    prediction_service: PredictionServiceDep,
    feature_validator: FeatureValidatorDep,
) -> Dict[str, Any]:
    """
    Make a single prediction with feature validation.

    This endpoint:
    1. Validates input features against the model's expected schema
    2. Offloads the blocking ML inference to a thread pool
    3. Returns prediction with any validation warnings

    Args:
        request: The prediction request containing feature values.
        prediction_service: Injected prediction service
        feature_validator: Injected feature validator

    Returns:
        A dictionary containing the prediction score, model version, and any warnings.

    Raises:
        HTTPException: 422 if feature validation fails, 500 if prediction fails.
    """
    with LogContext(feature_count=len(request.features)):
        # Validate features
        mode = _get_validation_mode(request.validation_mode)
        validation_result = feature_validator.validate(request.features, mode=mode)

        if not validation_result.is_valid:
            logger.warning(
                "feature_validation_failed",
                error_count=len(validation_result.errors),
                mode=mode.value,
            )
            return JSONResponse(
                status_code=422,
                content={
                    "detail": "Feature validation failed",
                    "validation_errors": [e.to_dict() for e in validation_result.errors],
                },
            )

        try:
            # Use validated features if available
            features_to_use = validation_result.validated_features or request.features

            # Create new request with validated features
            validated_request = APIPredictionRequest(
                features=features_to_use,
                model_version=request.model_version,
            )

            # Offload blocking ML inference to thread pool
            result = await asyncio.to_thread(
                prediction_service.predict_single,
                validated_request
            )

            # Add validation warnings to response
            if validation_result.warnings:
                result["validation_warnings"] = [
                    {"code": w.code, "message": w.message, "field": w.field}
                    for w in validation_result.warnings
                ]

            logger.info(
                "prediction_success",
                prediction=result.get("prediction"),
                model_version=result.get("model_version"),
                warning_count=len(validation_result.warnings),
            )
            return result
        except PredictionError as e:
            logger.error("prediction_failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.exception("prediction_unexpected_error")
            raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.post(
    "/explain",
    response_model=APIExplanationResponse,
    responses={
        422: {"model": APIValidationErrorResponse, "description": "Validation Error"},
    },
)
async def explain(
    request: APIPredictionRequest,
    prediction_service: PredictionServiceDep,
    feature_validator: FeatureValidatorDep,
) -> Dict[str, Any]:
    """
    Generate a real-time explanation for a single prediction.

    This endpoint uses SHAP values to explain which features contributed
    most to the prediction. Features are validated before explanation.

    Args:
        request: The prediction request containing feature values.
        prediction_service: Injected prediction service
        feature_validator: Injected feature validator

    Returns:
        A dictionary containing the model version and feature contributions.

    Raises:
        HTTPException: 422 if validation fails, 500 if explanation fails.
    """
    with LogContext(feature_count=len(request.features)):
        # Validate features
        mode = _get_validation_mode(request.validation_mode)
        validation_result = feature_validator.validate(request.features, mode=mode)

        if not validation_result.is_valid:
            logger.warning(
                "feature_validation_failed_explain",
                error_count=len(validation_result.errors),
            )
            return JSONResponse(
                status_code=422,
                content={
                    "detail": "Feature validation failed",
                    "validation_errors": [e.to_dict() for e in validation_result.errors],
                },
            )

        try:
            # Use validated features
            features_to_use = validation_result.validated_features or request.features
            validated_request = APIPredictionRequest(
                features=features_to_use,
                model_version=request.model_version,
            )

            # Offload blocking SHAP computation to thread pool
            result = await asyncio.to_thread(
                prediction_service.explain_single,
                validated_request
            )
            logger.info(
                "explanation_success",
                top_features=list(result.get("feature_contributions", {}).keys())[:3],
            )
            return result
        except PredictionError as e:
            logger.error("explanation_failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.exception("explanation_unexpected_error")
            raise HTTPException(
                status_code=500,
                detail="An unexpected error occurred during explanation"
            )


@app.get("/health")
async def health_check(
    prediction_service: PredictionServiceDep,
) -> Dict[str, Any]:
    """
    Health check endpoint for Kubernetes liveness/readiness probes.

    Args:
        prediction_service: Injected prediction service

    Returns:
        A dictionary containing the health status and component states.
    """
    return prediction_service.health_check()


@app.get("/features", response_model=APIFeatureInfoResponse)
async def get_feature_info(
    feature_validator: FeatureValidatorDep,
) -> Dict[str, Any]:
    """
    Get information about expected features.

    This endpoint returns the list of features the model expects,
    useful for debugging validation errors and building clients.

    Args:
        feature_validator: Injected feature validator

    Returns:
        Feature names, count, and optional specifications.
    """
    return feature_validator.get_feature_info()


@app.post("/validate")
async def validate_features(
    request: APIPredictionRequest,
    feature_validator: FeatureValidatorDep,
) -> Dict[str, Any]:
    """
    Validate features without making a prediction.

    This endpoint is useful for checking if features are valid
    before sending them for prediction.

    Args:
        request: The prediction request to validate.
        feature_validator: Injected feature validator

    Returns:
        Validation result with errors and warnings.
    """
    mode = _get_validation_mode(request.validation_mode)
    result = feature_validator.validate(request.features, mode=mode)
    return result.to_dict()


@app.post("/predict/batch")
async def predict_batch(
    requests: list[APIPredictionRequest],
    prediction_service: PredictionServiceDep,
    feature_validator: FeatureValidatorDep,
) -> list[Dict[str, Any]]:
    """
    Make batch predictions for multiple inputs.

    Each request is validated independently. If any validation fails in
    strict/lenient mode, the entire batch fails.

    Args:
        requests: A list of prediction requests.
        prediction_service: Injected prediction service
        feature_validator: Injected feature validator

    Returns:
        A list of prediction results in the same order as the requests.

    Raises:
        HTTPException: 422 if any validation fails, 500 if prediction fails.
    """
    with LogContext(batch_size=len(requests)):
        logger.info("batch_prediction_started")

        # Validate all requests first
        validation_errors = []

        for i, req in enumerate(requests):
            mode = _get_validation_mode(req.validation_mode)
            result = feature_validator.validate(req.features, mode=mode)
            if not result.is_valid:
                validation_errors.append({
                    "index": i,
                    "errors": [e.to_dict() for e in result.errors],
                })

        if validation_errors:
            logger.warning(
                "batch_validation_failed",
                failed_count=len(validation_errors),
            )
            return JSONResponse(
                status_code=422,
                content={
                    "detail": f"Validation failed for {len(validation_errors)} request(s)",
                    "validation_errors": validation_errors,
                },
            )

        try:
            # Process all predictions concurrently
            tasks = [
                asyncio.to_thread(prediction_service.predict_single, req)
                for req in requests
            ]
            results = await asyncio.gather(*tasks)
            logger.info("batch_prediction_success", result_count=len(results))
            return list(results)
        except PredictionError as e:
            logger.error("batch_prediction_failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.exception("batch_prediction_unexpected_error")
            raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.post("/reload")
async def reload_model(
    prediction_service: PredictionServiceDep,
) -> Dict[str, str]:
    """
    Reload the model from disk/registry.

    This endpoint is useful for updating the model without restarting the server.

    Args:
        prediction_service: Injected prediction service

    Returns:
        Status message
    """
    try:
        await asyncio.to_thread(prediction_service.reload)
        logger.info("model_reloaded")
        return {"status": "Model reloaded successfully"}
    except Exception as e:
        logger.error("model_reload_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {e}")
