"""Event bus implementations."""

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Any

from .models import Event
from ..utils.logging import get_logger

logger = get_logger(__name__)

EventHandler = Callable[[Event], Any]


class EventBus(ABC):
    """
    Event bus interface.

    Design Pattern: Observer + Strategy
    """

    @abstractmethod
    async def publish(self, event: Event) -> None:
        """Publish event to all subscribers."""
        pass

    @abstractmethod
    def subscribe(self, pattern: str, handler: EventHandler) -> str:
        """Subscribe to events matching pattern. Returns subscription ID."""
        pass

    @abstractmethod
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe handler."""
        pass


class AsyncIOEventBus(EventBus):
    """
    In-memory event bus using asyncio.

    Suitable for single-instance deployment.
    Thread-safe via asyncio primitives.
    """

    def __init__(self):
        self._handlers: dict[str, list[tuple[str, EventHandler]]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._counter = 0

    async def publish(self, event: Event) -> None:
        """Publish to matching handlers."""
        handlers = await self._get_matching_handlers(event.event_type.value)

        for _, handler in handlers:
            asyncio.create_task(self._safe_call(handler, event))

    def subscribe(self, pattern: str, handler: EventHandler) -> str:
        """Subscribe to events matching pattern."""
        self._counter += 1
        sub_id = f"bus_sub_{self._counter}"
        self._handlers[pattern].append((sub_id, handler))
        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Remove handler by subscription ID."""
        for pattern, handlers in self._handlers.items():
            for i, (sub_id, _) in enumerate(handlers):
                if sub_id == subscription_id:
                    handlers.pop(i)
                    return True
        return False

    async def _get_matching_handlers(
        self, event_type: str
    ) -> list[tuple[str, EventHandler]]:
        """Get handlers matching event type."""
        async with self._lock:
            matching = []
            for pattern, handlers in self._handlers.items():
                if self._pattern_matches(pattern, event_type):
                    matching.extend(handlers)
            return matching

    async def _safe_call(self, handler: EventHandler, event: Event) -> None:
        """Call handler with error isolation."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            logger.error(f"Event handler error: {e}", exc_info=True)

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
