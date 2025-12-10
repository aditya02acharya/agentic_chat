"""Base node class for PocketFlow integration."""

from abc import ABC, abstractmethod
from typing import Any

from pocketflow import AsyncNode

from agentic_chatbot.events.emitter import EventEmitter
from agentic_chatbot.events.models import Event
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class AsyncBaseNode(AsyncNode, ABC):
    """
    Base async node with event emission support.

    Extends PocketFlow's AsyncNode with:
    - Event emission for SSE streaming
    - Structured logging
    - Error handling utilities

    Design Pattern: Template Method
    """

    # Node metadata
    node_name: str = "base_node"
    description: str = "Base node"

    def __init__(self, **kwargs: Any):
        """Initialize node with optional parameters."""
        super().__init__(**kwargs)
        self._emitter: EventEmitter | None = None

    def _get_emitter(self, shared: dict[str, Any]) -> EventEmitter | None:
        """Get event emitter from shared store."""
        if self._emitter:
            return self._emitter

        # Try to get from shared
        emitter = shared.get("event_emitter")
        if emitter and isinstance(emitter, EventEmitter):
            return emitter

        # Try to create from queue
        queue = shared.get("event_queue")
        if queue:
            return EventEmitter(queue)

        return None

    async def emit_event(self, shared: dict[str, Any], event: Event) -> None:
        """
        Emit an event to the SSE stream.

        Args:
            shared: Shared store containing event emitter/queue
            event: Event to emit
        """
        emitter = self._get_emitter(shared)
        if emitter:
            await emitter.emit(event)
        else:
            logger.debug(f"No emitter available for event: {event.event_type}")

    async def prep_async(self, shared: dict[str, Any]) -> Any:
        """
        Prepare data for execution.

        Override this to read from shared store.

        Args:
            shared: Shared store

        Returns:
            Data for exec_async
        """
        return None

    @abstractmethod
    async def exec_async(self, prep_res: Any) -> Any:
        """
        Execute node logic.

        This is where the main work happens.
        Do NOT access shared store here.

        Args:
            prep_res: Result from prep_async

        Returns:
            Execution result
        """
        pass

    async def post_async(
        self,
        shared: dict[str, Any],
        prep_res: Any,
        exec_res: Any,
    ) -> str | None:
        """
        Post-process and store results.

        Override this to write to shared store and
        return the action for flow routing.

        Args:
            shared: Shared store
            prep_res: Result from prep_async
            exec_res: Result from exec_async

        Returns:
            Action string for flow routing, or None for default
        """
        return None

    async def exec_fallback_async(self, prep_res: Any, exc: Exception) -> Any:
        """
        Handle execution failure.

        Override this to provide graceful fallback.

        Args:
            prep_res: Result from prep_async
            exc: Exception that occurred

        Returns:
            Fallback result
        """
        logger.error(
            f"Node {self.node_name} failed",
            error=str(exc),
            exc_info=True,
        )
        raise exc
