"""Observability configuration using Langfuse.

Langfuse provides OTEL-based tracing for LLM applications.
It automatically integrates with LangChain/LangGraph.

Usage:
    1. Set environment variables:
       - LANGFUSE_PUBLIC_KEY
       - LANGFUSE_SECRET_KEY
       - LANGFUSE_HOST (optional, defaults to cloud.langfuse.com)

    2. Call configure_observability() at startup

    3. Use @observe decorator on functions you want to trace
"""

import os
from functools import wraps
from typing import Any, Callable, TypeVar

from agentic_chatbot.utils.logging import get_logger

logger = get_logger(__name__)

# Type variable for decorator
F = TypeVar("F", bound=Callable[..., Any])

# Global flag to track if Langfuse is configured
_langfuse_configured = False


def configure_observability() -> bool:
    """
    Configure Langfuse observability.

    Returns True if Langfuse was successfully configured, False otherwise.
    """
    global _langfuse_configured

    if _langfuse_configured:
        return True

    # Check for required environment variables
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        logger.info("Langfuse credentials not found, observability disabled")
        return False

    try:
        from langfuse import Langfuse

        # Initialize Langfuse client
        # This automatically configures the global client
        Langfuse()

        _langfuse_configured = True
        logger.info("Langfuse observability configured successfully")
        return True

    except ImportError:
        logger.warning("langfuse package not installed, observability disabled")
        return False
    except Exception as e:
        logger.warning(f"Failed to configure Langfuse: {e}")
        return False


def observe(name: str | None = None) -> Callable[[F], F]:
    """
    Decorator to trace a function with Langfuse.

    If Langfuse is not configured, this is a no-op decorator.

    Args:
        name: Optional name for the trace (defaults to function name)

    Usage:
        @observe("my_operation")
        async def my_function():
            ...
    """
    def decorator(func: F) -> F:
        # Try to import Langfuse's observe decorator
        try:
            from langfuse.decorators import observe as langfuse_observe

            # Apply Langfuse's observe decorator
            traced_func = langfuse_observe(name=name or func.__name__)(func)
            return traced_func  # type: ignore

        except ImportError:
            # Langfuse not available, return original function
            return func

    return decorator


def get_langfuse_handler():
    """
    Get a Langfuse callback handler for LangChain/LangGraph.

    Returns None if Langfuse is not configured.
    """
    try:
        from langfuse.callback import CallbackHandler

        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

        if not public_key or not secret_key:
            return None

        return CallbackHandler()

    except ImportError:
        return None
    except Exception as e:
        logger.warning(f"Failed to create Langfuse handler: {e}")
        return None


def flush_observability() -> None:
    """
    Flush any pending traces to Langfuse.

    Call this before shutdown to ensure all traces are sent.
    """
    try:
        from langfuse import get_client

        client = get_client()
        if client:
            client.flush()
            logger.debug("Flushed Langfuse traces")

    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to flush Langfuse: {e}")
