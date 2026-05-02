"""
YADEM Rate Limiting Middleware
=============================================================================
Redis-backed rate limiting per API key. Falls back to in-memory for MVP.
"""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
from datetime import datetime
import time
from loguru import logger


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token bucket rate limiter.
    Default: 100 requests per minute per API key.
    """

    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.rpm = requests_per_minute
        self._buckets: dict = defaultdict(lambda: {"tokens": requests_per_minute,
                                                    "last_refill": time.time()})

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ("/api/v1/health", "/docs", "/openapi.json", "/"):
            return await call_next(request)

        key = request.headers.get("X-API-Key", request.client.host)
        bucket = self._buckets[key]

        # Refill tokens
        now = time.time()
        elapsed = now - bucket["last_refill"]
        refill = int(elapsed * (self.rpm / 60.0))
        if refill > 0:
            bucket["tokens"] = min(self.rpm, bucket["tokens"] + refill)
            bucket["last_refill"] = now

        # Check tokens
        if bucket["tokens"] <= 0:
            logger.warning(f"Rate limit exceeded for key: {key[:12]}...")
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.rpm} requests/minute.",
                headers={"Retry-After": "60"},
            )

        bucket["tokens"] -= 1

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.rpm)
        response.headers["X-RateLimit-Remaining"] = str(bucket["tokens"])
        return response
