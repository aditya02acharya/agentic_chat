"""Event bus for application-wide event handling."""

import asyncio
import fnmatch
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Awaitable

from agentic_chatbot.events.models import Event
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


EventHandler = Callable[[Event], Awaitable[None]] | Callable[[Event], None]


class EventBus(ABC):
    """
    Event bus interface.

    Design Pattern: Observer + Strategy

    Current implementation: In-memory (AsyncIOEventBus)
    Future option: Redis-backed for multi-instance deployment
    """

    @abstractmethod
    async def publish(self, event: Event) -> None:
        """Publish event to all subscribers."""
        ...

    @abstractmethod
    def subscribe(self, pattern: str, handler: EventHandler) -> str:
        """Subscribe to events matching pattern. Returns subscription ID."""
        ...

    @abstractmethod
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe handler."""
        ...


class AsyncIOEventBus(EventBus):
    """
    In-memory event bus using asyncio.

    Suitable for single-instance deployment.
    Thread-safe via asyncio primitives.
    """

    def __init__(self):
        # pattern -> list of (subscription_id, handler)
        self._handlers: dict[str, list[tuple[str, EventHandler]]] = defaultdict(list)
        self._subscription_map: dict[str, str] = {}  # subscription_id -> pattern
        self._lock = asyncio.Lock()

    async def publish(self, event: Event) -> None:
        """Publish to matching handlers."""
        handlers = await self._get_matching_handlers(event.event_type.value)

        # Fire-and-forget for non-blocking
        for _, handler in handlers:
            asyncio.create_task(self._safe_call(handler, event))

    def subscribe(self, pattern: str, handler: EventHandler) -> str:
        """
        Subscribe to events matching pattern.

        Pattern supports wildcards:
        - "supervisor.*" matches supervisor.thinking, supervisor.decided
        - "tool.*" matches tool.start, tool.complete, etc.
        - "*" matches all events

        Returns subscription ID for unsubscribing.
        """
        subscription_id = str(uuid.uuid4())
        self._handlers[pattern].append((subscription_id, handler))
        self._subscription_map[subscription_id] = pattern
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe handler by ID."""
        if subscription_id not in self._subscription_map:
            return False

        pattern = self._subscription_map.pop(subscription_id)
        self._handlers[pattern] = [
            (sid, h) for sid, h in self._handlers[pattern] if sid != subscription_id
        ]
        return True

    async def _get_matching_handlers(self, event_type: str) -> list[tuple[str, EventHandler]]:
        """Get all handlers matching the event type."""
        async with self._lock:
            handlers = []
            for pattern, pattern_handlers in self._handlers.items():
                if fnmatch.fnmatch(event_type, pattern):
                    handlers.extend(pattern_handlers)
            return handlers

    async def _safe_call(self, handler: EventHandler, event: Event) -> None:
        """Call handler with error isolation."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            logger.error(f"Event handler error: {e}", exc_info=True)
