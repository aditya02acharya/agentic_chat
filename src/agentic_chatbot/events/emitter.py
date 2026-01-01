"""Event emitter for publishing events."""

import asyncio
from typing import Callable, Awaitable

from agentic_chatbot.events.models import Event
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)

EventHandler = Callable[[Event], Awaitable[None]] | Callable[[Event], None]


class EventEmitter:
    """
    Simple event emitter for request-scoped event handling.

    Design Pattern: Observer Pattern

    Each request gets its own EventEmitter instance that routes
    events to the SSE queue and any registered handlers.
    """

    def __init__(self, event_queue: asyncio.Queue[Event] | None = None):
        """
        Initialize event emitter.

        Args:
            event_queue: Optional queue for SSE streaming
        """
        self._queue = event_queue
        self._handlers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> None:
        """Subscribe a handler to receive all events."""
        self._handlers.append(handler)

    def unsubscribe(self, handler: EventHandler) -> None:
        """Unsubscribe a handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    async def emit(self, event: Event) -> None:
        """
        Emit an event to the queue and all handlers.

        Args:
            event: Event to emit
        """
        # Send to SSE queue if configured
        if self._queue is not None:
            await self._queue.put(event)

        # Notify all handlers
        for handler in self._handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                # Log error but don't let handler errors break event flow
                logger.error(
                    "Event handler error",
                    event_type=event.event_type,
                    error=str(e),
                    exc_info=True,
                )

    def emit_sync(self, event: Event) -> None:
        """
        Emit an event synchronously (non-blocking).

        Creates a task to handle async emission.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.emit(event))
        except RuntimeError:
            # No running loop, skip emission
            pass
