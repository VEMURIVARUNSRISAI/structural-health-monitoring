"""
app/main.py — application entry point.

uvicorn imports this file and runs the `app` object.
The lifespan() function runs setup before the server starts and
teardown after it stops. Later parts add model loading and DB connection here.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging_config import setup_logging

settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup (before the server accepts requests) ──
    setup_logging(settings.log_level)
    logger.info(f"Starting SHM API in {settings.app_env} mode")
    # Stage 2 adds: load YOLO models here
    # Stage 4 adds: connect to the database here
    logger.info("SHM API ready")
    yield
    # ── shutdown ──
    logger.info("Shutting down SHM API")


app = FastAPI(
    title="Structural Health Monitoring API",
    description="Autonomous defect detection for civil infrastructure.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "environment": settings.app_env, "version": "0.1.0"}