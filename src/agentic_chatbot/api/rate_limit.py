"""Rate limiting middleware for API endpoints.

Simple in-memory rate limiter with sliding window.
For production, consider using Redis-backed rate limiting.
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, Any

from fastapi import HTTPException, Request
from pydantic import BaseModel

from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    # Requests per window
    requests_per_minute: int = 60
    requests_per_hour: int = 1000

    # Burst allowance (requests allowed in quick succession)
    burst_limit: int = 10
    burst_window_seconds: float = 1.0

    # Cleanup interval (seconds)
    cleanup_interval: float = 300.0


@dataclass
class ClientState:
    """Tracks request history for a client."""

    # Timestamps of recent requests
    minute_requests: list[float] = field(default_factory=list)
    hour_requests: list[float] = field(default_factory=list)
    burst_requests: list[float] = field(default_factory=list)

    # Last cleanup time
    last_cleanup: float = field(default_factory=time.time)


class RateLimiter:
    """
    In-memory rate limiter with sliding window.

    Tracks requests per client (by IP or API key) and enforces:
    - Per-minute limits
    - Per-hour limits
    - Burst limits (prevent rapid-fire requests)

    Usage:
        limiter = RateLimiter()

        @app.post("/chat")
        async def chat(request: Request):
            await limiter.check(request)
            ...
    """

    def __init__(self, config: RateLimitConfig | None = None):
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self._config = config or RateLimitConfig()
        self._clients: dict[str, ClientState] = defaultdict(ClientState)
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    def _get_client_key(self, request: Request) -> str:
        """
        Get unique key for client.

        Uses X-Forwarded-For header if behind proxy, otherwise client IP.
        Can be extended to use API keys for authenticated requests.
        """
        # Check for API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key}"

        # Use forwarded IP if behind proxy
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take first IP (original client)
            return f"ip:{forwarded.split(',')[0].strip()}"

        # Fall back to direct client IP
        client = request.client
        if client:
            return f"ip:{client.host}"

        return "ip:unknown"

    async def check(self, request: Request) -> None:
        """
        Check if request is allowed.

        Args:
            request: FastAPI request

        Raises:
            HTTPException: 429 if rate limit exceeded
        """
        client_key = self._get_client_key(request)
        now = time.time()

        async with self._lock:
            state = self._clients[client_key]

            # Cleanup old entries
            self._cleanup_client(state, now)

            # Check burst limit
            if len(state.burst_requests) >= self._config.burst_limit:
                oldest_burst = state.burst_requests[0]
                if now - oldest_burst < self._config.burst_window_seconds:
                    logger.warning(
                        "Rate limit exceeded (burst)",
                        client=client_key,
                        limit=self._config.burst_limit,
                    )
                    raise HTTPException(
                        status_code=429,
                        detail="Too many requests. Please slow down.",
                        headers={"Retry-After": "1"},
                    )

            # Check per-minute limit
            if len(state.minute_requests) >= self._config.requests_per_minute:
                logger.warning(
                    "Rate limit exceeded (per-minute)",
                    client=client_key,
                    limit=self._config.requests_per_minute,
                )
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please wait before making more requests.",
                    headers={"Retry-After": "60"},
                )

            # Check per-hour limit
            if len(state.hour_requests) >= self._config.requests_per_hour:
                logger.warning(
                    "Rate limit exceeded (per-hour)",
                    client=client_key,
                    limit=self._config.requests_per_hour,
                )
                raise HTTPException(
                    status_code=429,
                    detail="Hourly rate limit exceeded. Please try again later.",
                    headers={"Retry-After": "3600"},
                )

            # Record this request
            state.burst_requests.append(now)
            state.minute_requests.append(now)
            state.hour_requests.append(now)

    def _cleanup_client(self, state: ClientState, now: float) -> None:
        """Remove expired entries from client state."""
        # Remove burst requests older than burst window
        burst_cutoff = now - self._config.burst_window_seconds
        state.burst_requests = [
            t for t in state.burst_requests if t > burst_cutoff
        ]

        # Remove minute requests older than 60 seconds
        minute_cutoff = now - 60.0
        state.minute_requests = [
            t for t in state.minute_requests if t > minute_cutoff
        ]

        # Remove hour requests older than 3600 seconds
        hour_cutoff = now - 3600.0
        state.hour_requests = [
            t for t in state.hour_requests if t > hour_cutoff
        ]

    async def cleanup_all(self) -> int:
        """
        Clean up stale client entries.

        Returns:
            Number of clients removed
        """
        now = time.time()
        removed = 0

        async with self._lock:
            stale_clients = []

            for client_key, state in self._clients.items():
                # If no activity in the last hour, remove
                if not state.hour_requests:
                    stale_clients.append(client_key)
                elif max(state.hour_requests) < now - 3600:
                    stale_clients.append(client_key)

            for client_key in stale_clients:
                del self._clients[client_key]
                removed += 1

        if removed > 0:
            logger.debug(f"Cleaned up {removed} stale rate limit entries")

        return removed

    async def start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.debug("Started rate limiter cleanup task")

    async def stop_cleanup_task(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.debug("Stopped rate limiter cleanup task")

    async def _cleanup_loop(self) -> None:
        """Background task to periodically clean up stale entries."""
        while True:
            try:
                await asyncio.sleep(self._config.cleanup_interval)
                await self.cleanup_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rate limiter cleanup error: {e}")

    def get_client_stats(self, request: Request) -> dict[str, Any]:
        """Get rate limit stats for a client (for debugging)."""
        client_key = self._get_client_key(request)
        state = self._clients.get(client_key)

        if not state:
            return {
                "client": client_key,
                "minute_requests": 0,
                "hour_requests": 0,
                "burst_requests": 0,
            }

        return {
            "client": client_key,
            "minute_requests": len(state.minute_requests),
            "hour_requests": len(state.hour_requests),
            "burst_requests": len(state.burst_requests),
            "limits": {
                "per_minute": self._config.requests_per_minute,
                "per_hour": self._config.requests_per_hour,
                "burst": self._config.burst_limit,
            },
        }


# Default rate limiter instance
_default_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Get the default rate limiter instance."""
    global _default_limiter
    if _default_limiter is None:
        _default_limiter = RateLimiter()
    return _default_limiter


def rate_limit(func: Callable) -> Callable:
    """
    Decorator to apply rate limiting to an endpoint.

    Usage:
        @app.post("/chat")
        @rate_limit
        async def chat(request: Request, ...):
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Find the Request object in args or kwargs
        request = kwargs.get("request")
        if request is None:
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

        if request:
            limiter = get_rate_limiter()
            await limiter.check(request)

        return await func(*args, **kwargs)

    return wrapper
