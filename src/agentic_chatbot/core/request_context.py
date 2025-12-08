"""Request-scoped context with proper lifecycle management."""

import asyncio
from typing import Any, Callable, Awaitable
from uuid import uuid4

from ..events.models import Event
from ..utils.logging import get_logger

logger = get_logger(__name__)


class RequestContext:
    """
    Request-scoped context with proper lifecycle management.

    Features:
    - __slots__ prevents accidental attribute addition
    - Cleanup callbacks for resource management
    - Context manager support
    """

    __slots__ = (
        "conversation_id",
        "request_id",
        "user_query",
        "event_queue",
        "shared_store",
        "_cleanup_callbacks",
        "_closed",
    )

    def __init__(self, conversation_id: str, user_query: str):
        self.conversation_id = conversation_id
        self.request_id = str(uuid4())
        self.user_query = user_query
        self.event_queue: asyncio.Queue[Event] = asyncio.Queue()
        self.shared_store: dict[str, Any] = {}
        self._cleanup_callbacks: list[Callable[[], Awaitable[None]]] = []
        self._closed = False

    def register_cleanup(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register cleanup callback."""
        if self._closed:
            raise RuntimeError("Context already closed")
        self._cleanup_callbacks.append(callback)

    async def emit_event(self, event: Event) -> None:
        """Emit event to the queue."""
        event.request_id = self.request_id
        await self.event_queue.put(event)

    async def cleanup(self) -> None:
        """Run all cleanup callbacks. Idempotent."""
        if self._closed:
            return
        self._closed = True

        for callback in reversed(self._cleanup_callbacks):
            try:
                await callback()
            except Exception as e:
                logger.error(f"Cleanup error: {e}", exc_info=True)

        self._cleanup_callbacks.clear()
        self.shared_store.clear()

        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def __aenter__(self) -> "RequestContext":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.cleanup()
