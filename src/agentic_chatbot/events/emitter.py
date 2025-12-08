"""Event emitter for publishing events."""

import asyncio
from typing import Callable, Any

from .models import Event
from .types import EventType
from ..utils.logging import get_logger

logger = get_logger(__name__)

EventHandler = Callable[[Event], Any]


class EventEmitter:
    """
    Publish events to subscribers.

    Design Pattern: Observer Pattern

    Used to emit SSE events for real-time progress updates.
    """

    def __init__(self):
        self._handlers: dict[str, list[tuple[str, EventHandler]]] = {}
        self._handler_counter = 0

    def subscribe(self, pattern: str, handler: EventHandler) -> str:
        """
        Subscribe to events matching pattern.

        Args:
            pattern: Event type pattern (e.g., "tool.*", "response.chunk")
            handler: Callback function to handle events

        Returns:
            Subscription ID for unsubscribing
        """
        self._handler_counter += 1
        sub_id = f"sub_{self._handler_counter}"

        if pattern not in self._handlers:
            self._handlers[pattern] = []
        self._handlers[pattern].append((sub_id, handler))

        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe a handler by subscription ID."""
        for pattern, handlers in self._handlers.items():
            for i, (sub_id, _) in enumerate(handlers):
                if sub_id == subscription_id:
                    handlers.pop(i)
                    return True
        return False

    def emit(self, event: Event) -> None:
        """Emit event synchronously to all matching handlers."""
        handlers = self._get_matching_handlers(event.event_type)
        for _, handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                logger.error(f"Event handler error: {e}", exc_info=True)

    async def emit_async(self, event: Event) -> None:
        """Emit event asynchronously to all matching handlers."""
        handlers = self._get_matching_handlers(event.event_type)
        for _, handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}", exc_info=True)

    def _get_matching_handlers(
        self, event_type: EventType
    ) -> list[tuple[str, EventHandler]]:
        """Get handlers matching the event type."""
        matching = []
        event_str = event_type.value

        for pattern, handlers in self._handlers.items():
            if self._pattern_matches(pattern, event_str):
                matching.extend(handlers)

        return matching

    def _pattern_matches(self, pattern: str, event_type: str) -> bool:
        """Check if pattern matches event type."""
        if pattern == "*":
            return True
        if pattern == event_type:
            return True
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return event_type.startswith(prefix + ".")
        return False
