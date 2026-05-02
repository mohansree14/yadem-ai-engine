"""
YADEM Health Check Route
GET /api/v1/health
"""

from fastapi import APIRouter
from src.api.schemas.application import HealthResponse
import time

router = APIRouter()

_start_time = time.time()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model status."""
    from src.api.main import engine_state
    return HealthResponse(
        status="healthy",
        models_loaded=engine_state.get("models_loaded", False),
        version="1.0.0",
        uptime_seconds=round(time.time() - _start_time, 2),
    )
