"""FastAPI server for the Geopolitical Data Generator platform.

Provides REST endpoints for scenario execution, export, calibration, and
real-time WebSocket progress streaming.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.scenarios import router as scenarios_router
from api.routes.jobs import router as jobs_router
from api.routes.exports import router as exports_router
from api.routes.calibration import router as calibration_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    # Startup: initialize services, connect to Redis, load models
    logger.info("Geopolitical Data Generator API starting up...")
    yield
    # Shutdown: close connections, cleanup
    logger.info("Geopolitical Data Generator API shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Geopolitical Data Generator API",
        version="0.1.0",
        description=(
            "Causal geopolitical scenario simulation platform with "
            "Bayesian calibration, distributed execution, and real-time dashboards."
        ),
        lifespan=lifespan,
    )

    # CORS - configure for production origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(scenarios_router, prefix="/api/v1/scenarios", tags=["scenarios"])
    app.include_router(jobs_router, prefix="/api/v1/jobs", tags=["jobs"])
    app.include_router(exports_router, prefix="/api/v1/exports", tags=["exports"])
    app.include_router(
        calibration_router, prefix="/api/v1/calibration", tags=["calibration"]
    )

    @app.get("/", include_in_schema=False)
    async def root():
        """Root health check."""
        return {"status": "ok", "service": "geopolitical-data-generator"}

    return app


# Create app instance for `uvicorn api.main:create_app()` usage
app = create_app()
