"""
YADEM API Key Authentication Middleware
=============================================================================
Validates API keys on every request to protected endpoints.
Partners receive unique API keys via the YADEM dashboard.
"""

from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Set
from loguru import logger
import os
import time


API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# In production, these would be stored in a database
# For MVP, loaded from environment variable (comma-separated)
_VALID_API_KEYS: Set[str] = set()


def _load_api_keys():
    """Load valid API keys from environment."""
    global _VALID_API_KEYS
    keys_str = os.environ.get("YADEM_API_KEYS", "ydm_dev_key_12345,ydm_test_key_67890")
    _VALID_API_KEYS = {k.strip() for k in keys_str.split(",") if k.strip()}
    logger.info(f"Loaded {len(_VALID_API_KEYS)} API keys")


_load_api_keys()

# Paths that don't require authentication
PUBLIC_PATHS = {"/api/v1/health", "/docs", "/openapi.json", "/redoc", "/"}


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """FastAPI dependency to verify API key."""
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key. Set X-API-Key header.")
    if api_key not in _VALID_API_KEYS:
        logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware that validates API keys on all non-public routes."""

    async def dispatch(self, request: Request, call_next):
        # Skip auth for public paths
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Skip auth for OPTIONS (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing X-API-Key header")

        if api_key not in _VALID_API_KEYS:
            logger.warning(f"Unauthorized access attempt from {request.client.host}")
            raise HTTPException(status_code=403, detail="Invalid API key")

        # Attach partner info to request state
        request.state.api_key = api_key
        request.state.authenticated = True

        response = await call_next(request)
        return response
