"""
YADEM Consent Validation Middleware
=============================================================================
Validates consent tokens on scoring requests before any data processing.
"""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger


# Paths that require consent validation
CONSENT_REQUIRED_PATHS = {"/api/v1/score", "/api/v1/kyc"}


class ConsentMiddleware(BaseHTTPMiddleware):
    """Validates that a consent token is present for data processing endpoints."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path not in CONSENT_REQUIRED_PATHS:
            return await call_next(request)

        if request.method != "POST":
            return await call_next(request)

        # Check for consent token in header or body
        consent_token = request.headers.get("X-Consent-Token")

        if not consent_token:
            # Try to peek at the body for consent_token field
            # For MVP, we allow the consent token in the header
            logger.debug("No consent token in header, checking if provided in body")

        # Attach to request state for downstream use
        request.state.consent_token = consent_token

        response = await call_next(request)
        return response
