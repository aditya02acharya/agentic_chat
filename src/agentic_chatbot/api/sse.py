"""SSE streaming helpers."""

import asyncio
from typing import AsyncIterator

from ..core.request_context import RequestContext
from ..events.types import EventType
from ..utils.logging import get_logger

logger = get_logger(__name__)


async def event_stream(ctx: RequestContext, timeout: float = 60.0) -> AsyncIterator[str]:
    """
    Generate SSE events from request context.

    Args:
        ctx: Request context with event queue
        timeout: Timeout for waiting on events

    Yields:
        SSE formatted event strings
    """
    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    ctx.event_queue.get(),
                    timeout=timeout,
                )
                yield event.to_sse()

                if event.event_type == EventType.RESPONSE_DONE:
                    break

            except asyncio.TimeoutError:
                yield ": keepalive\n\n"

    except asyncio.CancelledError:
        logger.debug(f"Event stream cancelled for request {ctx.request_id}")
        raise


async def format_sse_event(event_type: str, data: str) -> str:
    """Format a single SSE event."""
    return f"event: {event_type}\ndata: {data}\n\n"
