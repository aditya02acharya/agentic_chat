"""SSE stream helpers."""

import asyncio
from typing import AsyncIterator

from agentic_chatbot.events.models import Event
from agentic_chatbot.events.types import EventType


async def event_generator(
    event_queue: asyncio.Queue[Event],
    timeout: float = 60.0,
) -> AsyncIterator[str]:
    """
    Generate SSE events from queue.

    Args:
        event_queue: Queue of events to stream
        timeout: Timeout for waiting on queue (seconds)

    Yields:
        SSE formatted event strings
    """
    while True:
        try:
            event = await asyncio.wait_for(event_queue.get(), timeout=timeout)
            yield event.to_sse()

            # Check for terminal events
            if event.event_type in (EventType.RESPONSE_DONE, EventType.ERROR):
                break

        except asyncio.TimeoutError:
            # Send keepalive comment
            yield ": keepalive\n\n"


async def event_generator_with_task(
    event_queue: asyncio.Queue[Event],
    task: asyncio.Task,
    timeout: float = 60.0,
) -> AsyncIterator[str]:
    """
    Generate SSE events while running a background task.

    Args:
        event_queue: Queue of events to stream
        task: Background task running the flow
        timeout: Timeout between events

    Yields:
        SSE formatted event strings
    """
    try:
        while not task.done():
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=timeout)
                yield event.to_sse()

                if event.event_type in (EventType.RESPONSE_DONE, EventType.ERROR):
                    break

            except asyncio.TimeoutError:
                # Send keepalive if task is still running
                if not task.done():
                    yield ": keepalive\n\n"

        # Drain any remaining events
        while not event_queue.empty():
            try:
                event = event_queue.get_nowait()
                yield event.to_sse()
            except asyncio.QueueEmpty:
                break

    finally:
        # Ensure task is cancelled if we exit early
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
