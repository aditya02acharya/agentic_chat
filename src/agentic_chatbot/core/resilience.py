"""Centralized resilience patterns using hyx.

This module provides pre-configured resilience patterns for the agentic chatbot:
- Retry with exponential backoff for transient failures
- Circuit breakers to prevent cascade failures
- Timeouts to bound operation duration
- Bulkheads to limit concurrent operations
- Fallbacks for graceful degradation

Usage:
    from agentic_chatbot.core.resilience import (
        mcp_retry,
        llm_retry,
        mcp_circuit_breaker,
        llm_circuit_breaker,
        operation_timeout,
    )

    @mcp_retry
    @mcp_circuit_breaker
    async def call_mcp_tool(...):
        ...

    @llm_retry
    @llm_circuit_breaker
    @operation_timeout(60.0)
    async def call_llm_api(...):
        ...
"""

from functools import wraps
from typing import Any, Callable, TypeVar

from hyx.retry.api import retry
from hyx.retry.backoffs import expo
from hyx.circuitbreaker.api import consecutive_breaker
from hyx.circuitbreaker.exceptions import BreakerFailing

# Alias for clarity
BreakerOpen = BreakerFailing
from hyx.timeout.api import timeout
from hyx.timeout.exceptions import MaxDurationExceeded

# Alias for clarity
MaxTimeoutExceeded = MaxDurationExceeded
from hyx.bulkhead.api import bulkhead
from hyx.bulkhead.exceptions import BulkheadFull
from hyx.fallback.api import fallback

# Re-export exceptions for use by callers
__all__ = [
    # Exceptions
    "BreakerOpen",
    "MaxTimeoutExceeded",
    "BulkheadFull",
    "TransientError",
    "RateLimitError",
    # MCP patterns
    "mcp_retry",
    "mcp_circuit_breaker",
    "mcp_bulkhead",
    # LLM patterns
    "llm_retry",
    "llm_circuit_breaker",
    "llm_timeout",
    # Generic patterns
    "operation_timeout",
    "with_fallback",
    # Configuration
    "ResilienceConfig",
]


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================


class TransientError(Exception):
    """Error that is likely to succeed on retry (network issues, timeouts)."""
    pass


class RateLimitError(Exception):
    """Error indicating rate limiting (HTTP 429, throttling)."""
    pass


# =============================================================================
# CONFIGURATION
# =============================================================================


class ResilienceConfig:
    """Centralized configuration for resilience patterns."""

    # MCP Configuration
    MCP_RETRY_ATTEMPTS: int = 3
    MCP_RETRY_BACKOFF_BASE: float = 1.0  # seconds
    MCP_RETRY_BACKOFF_MAX: float = 30.0  # seconds

    MCP_CIRCUIT_FAILURE_THRESHOLD: int = 5
    MCP_CIRCUIT_RECOVERY_TIME: float = 30.0  # seconds
    MCP_CIRCUIT_RECOVERY_THRESHOLD: int = 2

    MCP_TIMEOUT: float = 30.0  # seconds
    MCP_BULKHEAD_CONCURRENCY: int = 10
    MCP_BULKHEAD_CAPACITY: int = 20

    # LLM Configuration
    LLM_RETRY_ATTEMPTS: int = 3
    LLM_RETRY_BACKOFF_BASE: float = 2.0  # seconds (longer for rate limits)
    LLM_RETRY_BACKOFF_MAX: float = 60.0  # seconds

    LLM_CIRCUIT_FAILURE_THRESHOLD: int = 3
    LLM_CIRCUIT_RECOVERY_TIME: float = 60.0  # seconds (longer for API issues)
    LLM_CIRCUIT_RECOVERY_THRESHOLD: int = 1

    LLM_TIMEOUT: float = 120.0  # seconds (LLM calls can be slow)

    # Registry Configuration
    REGISTRY_RETRY_ATTEMPTS: int = 3
    REGISTRY_TIMEOUT: float = 10.0  # seconds
    REGISTRY_CIRCUIT_FAILURE_THRESHOLD: int = 3
    REGISTRY_CIRCUIT_RECOVERY_TIME: float = 60.0  # seconds


# =============================================================================
# MCP RESILIENCE PATTERNS
# =============================================================================


# Retry for MCP calls with exponential backoff
mcp_retry = retry(
    on=(TransientError, ConnectionError, TimeoutError, OSError),
    attempts=ResilienceConfig.MCP_RETRY_ATTEMPTS,
    backoff=expo(
        min_delay_secs=ResilienceConfig.MCP_RETRY_BACKOFF_BASE,
        max_delay_secs=ResilienceConfig.MCP_RETRY_BACKOFF_MAX,
    ),
)

# Circuit breaker for MCP servers
mcp_circuit_breaker = consecutive_breaker(
    exceptions=(TransientError, ConnectionError, TimeoutError, BreakerOpen),
    failure_threshold=ResilienceConfig.MCP_CIRCUIT_FAILURE_THRESHOLD,
    recovery_time_secs=ResilienceConfig.MCP_CIRCUIT_RECOVERY_TIME,
    recovery_threshold=ResilienceConfig.MCP_CIRCUIT_RECOVERY_THRESHOLD,
)

# Bulkhead for MCP concurrency control
mcp_bulkhead = bulkhead(
    max_concurrency=ResilienceConfig.MCP_BULKHEAD_CONCURRENCY,
    max_capacity=ResilienceConfig.MCP_BULKHEAD_CAPACITY,
)


# =============================================================================
# LLM RESILIENCE PATTERNS
# =============================================================================


# Retry for LLM API calls with exponential backoff
llm_retry = retry(
    on=(TransientError, RateLimitError, ConnectionError, TimeoutError),
    attempts=ResilienceConfig.LLM_RETRY_ATTEMPTS,
    backoff=expo(
        min_delay_secs=ResilienceConfig.LLM_RETRY_BACKOFF_BASE,
        max_delay_secs=ResilienceConfig.LLM_RETRY_BACKOFF_MAX,
    ),
)

# Circuit breaker for LLM providers
llm_circuit_breaker = consecutive_breaker(
    exceptions=(TransientError, RateLimitError, ConnectionError),
    failure_threshold=ResilienceConfig.LLM_CIRCUIT_FAILURE_THRESHOLD,
    recovery_time_secs=ResilienceConfig.LLM_CIRCUIT_RECOVERY_TIME,
    recovery_threshold=ResilienceConfig.LLM_CIRCUIT_RECOVERY_THRESHOLD,
)

# Type variable for generic functions
F = TypeVar("F", bound=Callable[..., Any])


# Timeout for LLM operations - use lazy wrapper to avoid event loop issues at import time
def llm_timeout(func: F) -> F:
    """
    Apply timeout to LLM operations with lazy initialization.

    This wrapper defers timeout manager creation to runtime when
    an event loop is available, unlike the raw hyx.timeout decorator.
    """
    import asyncio
    from functools import wraps

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=ResilienceConfig.LLM_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise MaxTimeoutExceeded(
                f"Operation timed out after {ResilienceConfig.LLM_TIMEOUT}s"
            )

    return wrapper  # type: ignore


# =============================================================================
# GENERIC PATTERNS
# =============================================================================


def operation_timeout(max_seconds: float):
    """Create a timeout decorator with custom duration."""
    return timeout(max_delay_secs=max_seconds)


def with_fallback(fallback_value: Any, on: tuple = (Exception,)):
    """
    Create a fallback decorator that returns a default value on failure.

    Args:
        fallback_value: Value to return on failure
        on: Exception types to catch

    Usage:
        @with_fallback(fallback_value=[], on=(ConnectionError,))
        async def get_items():
            ...
    """
    async def fallback_handler(*args: Any, **kwargs: Any) -> Any:
        return fallback_value

    return fallback(handler=fallback_handler, on=on)


# =============================================================================
# HELPER DECORATORS
# =============================================================================


def classify_http_error(status_code: int) -> Exception:
    """
    Classify HTTP status codes into appropriate exceptions.

    Args:
        status_code: HTTP status code

    Returns:
        Appropriate exception type

    Raises:
        TransientError: For 5xx errors and 429 (rate limit)
        RateLimitError: For 429 specifically
        Exception: For other errors
    """
    if status_code == 429:
        return RateLimitError(f"Rate limited (HTTP {status_code})")
    elif status_code >= 500:
        return TransientError(f"Server error (HTTP {status_code})")
    elif status_code >= 400:
        return Exception(f"Client error (HTTP {status_code})")
    return Exception(f"Unknown error (HTTP {status_code})")


def wrap_httpx_errors(func: F) -> F:
    """
    Decorator to convert httpx exceptions to resilience-aware exceptions.

    This allows the retry and circuit breaker patterns to properly
    classify transient vs permanent errors.
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        import httpx

        try:
            return await func(*args, **kwargs)
        except httpx.TimeoutException as e:
            raise TransientError(f"Request timeout: {e}") from e
        except httpx.ConnectError as e:
            raise TransientError(f"Connection error: {e}") from e
        except httpx.HTTPStatusError as e:
            raise classify_http_error(e.response.status_code) from e

    return wrapper  # type: ignore


def wrap_anthropic_errors(func: F) -> F:
    """
    Decorator to convert Anthropic API exceptions to resilience-aware exceptions.
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            import anthropic
        except ImportError:
            return await func(*args, **kwargs)

        try:
            return await func(*args, **kwargs)
        except anthropic.RateLimitError as e:
            raise RateLimitError(f"Anthropic rate limit: {e}") from e
        except anthropic.APIConnectionError as e:
            raise TransientError(f"Anthropic connection error: {e}") from e
        except anthropic.InternalServerError as e:
            raise TransientError(f"Anthropic server error: {e}") from e
        except anthropic.APIStatusError as e:
            if e.status_code == 529:  # Overloaded
                raise TransientError(f"Anthropic overloaded: {e}") from e
            raise

    return wrapper  # type: ignore


def wrap_aws_errors(func: F) -> F:
    """
    Decorator to convert AWS/Bedrock exceptions to resilience-aware exceptions.
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            from botocore.exceptions import ClientError
        except ImportError:
            return await func(*args, **kwargs)

        try:
            return await func(*args, **kwargs)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("ThrottlingException", "TooManyRequestsException"):
                raise RateLimitError(f"AWS throttled: {e}") from e
            elif error_code in ("ServiceUnavailableException", "InternalServerError"):
                raise TransientError(f"AWS service error: {e}") from e
            raise

    return wrapper  # type: ignore


# =============================================================================
# COMPOSED PATTERNS
# =============================================================================


def mcp_resilient(func: F) -> F:
    """
    Apply full MCP resilience stack to a function.

    Applies (in order):
    1. Retry with exponential backoff
    2. Circuit breaker
    3. Timeout
    4. HTTP error classification

    Usage:
        @mcp_resilient
        async def call_mcp_server(...):
            ...
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await func(*args, **kwargs)

    # Apply decorators in reverse order (innermost first)
    wrapped = wrap_httpx_errors(wrapper)
    wrapped = timeout(max_delay_secs=ResilienceConfig.MCP_TIMEOUT)(wrapped)
    wrapped = mcp_circuit_breaker(wrapped)
    wrapped = mcp_retry(wrapped)

    return wrapped  # type: ignore


def llm_resilient(func: F) -> F:
    """
    Apply full LLM resilience stack to a function.

    Applies (in order):
    1. Retry with exponential backoff
    2. Circuit breaker
    3. Timeout
    4. Provider-specific error classification

    Usage:
        @llm_resilient
        async def call_llm_api(...):
            ...
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await func(*args, **kwargs)

    # Apply decorators in reverse order (innermost first)
    wrapped = wrap_anthropic_errors(wrapper)
    wrapped = llm_timeout(wrapped)
    wrapped = llm_circuit_breaker(wrapped)
    wrapped = llm_retry(wrapped)

    return wrapped  # type: ignore
