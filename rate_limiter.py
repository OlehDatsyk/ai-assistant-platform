"""
rate_limiter.py
Lightweight in-memory sliding-window rate limiter middleware.
Good enough for a single-process portfolio deployment; swap for Redis in prod.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.window_seconds = 60
        self.limit = settings.rate_limit_per_minute

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/static"):
            return await call_next(request)

        client_id = request.client.host if request.client else "unknown"
        now = time.time()
        bucket = self.hits[client_id]

        while bucket and now - bucket[0] > self.window_seconds:
            bucket.popleft()

        if len(bucket) >= self.limit:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please slow down."},
            )

        bucket.append(now)
        return await call_next(request)
