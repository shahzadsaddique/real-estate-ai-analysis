"""
FastAPI application entry point.

This module initializes the FastAPI application with:
- CORS configuration
- Middleware (request ID, logging)
- Error handlers
- Health check endpoint
- API router mounting
- Startup/shutdown event handlers
- Structured logging
"""

import logging
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Callable

import uvicorn
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import settings

# Configure structured logging
# Note: request_id will be added via custom log record factory
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

# Set default request_id for log records before middleware runs
old_factory = logging.getLogRecordFactory()


def default_record_factory(*args, **kwargs):
    """Default log record factory with request_id."""
    record = old_factory(*args, **kwargs)
    if not hasattr(record, "request_id"):
        record.request_id = "system"
    return record


logging.setLogRecordFactory(default_record_factory)

# Create logger
logger = logging.getLogger(__name__)


# ============================================================================
# Response Models
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    service: str
    version: str = "1.0.0"
    timestamp: str

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "service": "realestate-api",
                "version": "1.0.0",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: str
    request_id: str
    timestamp: str

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "Invalid input provided",
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        }


# ============================================================================
# Request ID Middleware
# ============================================================================


class RequestIDMiddleware:
    """Middleware to add request ID to all requests for tracing."""

    async def __call__(
        self, request: Request, call_next: Callable[[Request], Any]
    ) -> Response:
        """Add request ID to request and response headers."""
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Add request ID to request state for logging
        request.state.request_id = request_id

        # Add request ID to logger context
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.request_id = getattr(request.state, "request_id", "unknown")
            return record

        logging.setLogRecordFactory(record_factory)

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response


# ============================================================================
# Startup and Shutdown Handlers
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    Handles:
    - Application startup (database connections, service initialization)
    - Application shutdown (cleanup, graceful shutdown)
    """
    # Startup
    # Temporarily set request_id to "startup" for startup logs
    old_factory = logging.getLogRecordFactory()
    
    def startup_record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.request_id = "startup"
        return record
    
    logging.setLogRecordFactory(startup_record_factory)
    
    logger.info("Starting Real Estate AI Analysis Platform API")
    logger.info(f"Environment: {settings.gcp_region}")
    logger.info(f"Log Level: {settings.log_level}")
    
    # Restore default factory
    logging.setLogRecordFactory(default_record_factory)

    # Initialize services at startup
    from services import firestore_service, pubsub_service, storage_service
    
    try:
        # Initialize services (they check if already initialized)
        storage_service.initialize()
        pubsub_service.initialize()
        await firestore_service.initialize()
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}", exc_info=True)
        # Don't raise - allow app to start but services will initialize on first use

    yield

    # Shutdown
    # Temporarily set request_id to "shutdown" for shutdown logs
    old_factory = logging.getLogRecordFactory()
    
    def shutdown_record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.request_id = "shutdown"
        return record
    
    logging.setLogRecordFactory(shutdown_record_factory)
    
    logger.info("Shutting down Real Estate AI Analysis Platform API")
    
    # Restore default factory
    logging.setLogRecordFactory(default_record_factory)

    # Cleanup services
    from services import firestore_service, pubsub_service, storage_service
    
    try:
        await firestore_service.close()
        pubsub_service.close()
        storage_service.close()
        logger.info("All services closed successfully")
    except Exception as e:
        logger.error(f"Error during service cleanup: {str(e)}", exc_info=True)


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Real Estate AI Analysis Platform API",
    description="Production-ready API for real estate document analysis using LLMs",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ============================================================================
# Middleware Configuration
# ============================================================================
# Note: Middleware order matters. CORS should be added first (runs last in stack)
# to ensure it can handle OPTIONS preflight requests before other middleware

# Request ID Middleware - runs first (after CORS)
app.middleware("http")(RequestIDMiddleware())

# CORS Middleware - runs last (handles OPTIONS preflight)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
    max_age=3600,  # Cache preflight for 1 hour
)


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")

    # The middleware should have already set the factory to use request.state.request_id
    # If not, the default factory will use "system"
    logger.error(
        f"Unhandled exception: {str(exc)}",
        exc_info=True,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "detail": "An internal server error occurred",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handler for ValueError exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")

    # The middleware should have already set the factory to use request.state.request_id
    # If not, the default factory will use "system"
    logger.warning(f"ValueError: {str(exc)}")

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "ValidationError",
            "detail": str(exc),
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# ============================================================================
# Health Check Endpoint
# ============================================================================


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Health Check",
    description="Check the health status of the API service",
)
async def health_check(request: Request) -> HealthResponse:
    """
    Health check endpoint.

    Returns the current health status of the API service.
    Useful for load balancer health checks and monitoring.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    # The request_id is already set by the RequestIDMiddleware
    logger.debug("Health check requested")

    return HealthResponse(
        status="healthy",
        service="realestate-api",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ============================================================================
# API Router Mounting
# ============================================================================

from api.routes import analysis, documents

app.include_router(
    documents.router,
    prefix="/api/v1/documents",
    tags=["Documents"],
)

app.include_router(
    analysis.router,
    prefix="/api/v1",
    tags=["Analysis"],
)


# ============================================================================
# Root Endpoint
# ============================================================================


@app.get("/", tags=["Root"])
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "message": "Real Estate AI Analysis Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
    )
