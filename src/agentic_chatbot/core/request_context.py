"""Request-scoped context with proper lifecycle management."""

import asyncio
from typing import Any, Callable, Awaitable
from uuid import uuid4

from agentic_chatbot.events.models import Event
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class RequestContext:
    """
    Request-scoped context with proper lifecycle management.

    Features:
    - __slots__ prevents accidental attribute addition
    - Cleanup callbacks for resource management
    - Context manager support

    Design Pattern: Context Manager for resource lifecycle
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

    def __init__(
        self,
        conversation_id: str,
        user_query: str,
        request_id: str | None = None,
    ):
        """
        Initialize request context.

        Args:
            conversation_id: Conversation identifier
            user_query: User's query text
            request_id: Optional request ID (generated if not provided)
        """
        self.conversation_id = conversation_id
        self.request_id = request_id or str(uuid4())
        self.user_query = user_query
        self.event_queue: asyncio.Queue[Event] = asyncio.Queue()
        self.shared_store: dict[str, Any] = {}
        self._cleanup_callbacks: list[Callable[[], Awaitable[None]]] = []
        self._closed = False

    def register_cleanup(self, callback: Callable[[], Awaitable[None]]) -> None:
        """
        Register cleanup callback.

        Args:
            callback: Async function to call during cleanup
        """
        if self._closed:
            raise RuntimeError("Context already closed")
        self._cleanup_callbacks.append(callback)

    async def cleanup(self) -> None:
        """Run all cleanup callbacks. Idempotent."""
        if self._closed:
            return
        self._closed = True

        logger.debug("Cleaning up request context", request_id=self.request_id)

        # Run callbacks in reverse order (LIFO)
        for callback in reversed(self._cleanup_callbacks):
            try:
                await callback()
            except Exception as e:
                logger.error(f"Cleanup error: {e}", exc_info=True)

        self._cleanup_callbacks.clear()
        self.shared_store.clear()

        # Drain event queue
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def __aenter__(self) -> "RequestContext":
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context with cleanup."""
        await self.cleanup()

    @property
    def is_closed(self) -> bool:
        """Check if context is closed."""
        return self._closed

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from shared store."""
        return self.shared_store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in shared store."""
        self.shared_store[key] = value

    def update(self, data: dict[str, Any]) -> None:
        """Update shared store with multiple values."""
        self.shared_store.update(data)
