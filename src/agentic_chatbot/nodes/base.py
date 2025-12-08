"""Base node class for flow execution."""

from abc import ABC, abstractmethod
from typing import Any

from ..events.models import Event
from ..events.types import EventType
from ..core.request_context import RequestContext
from ..utils.logging import get_logger

logger = get_logger(__name__)


class AsyncBaseNode(ABC):
    """
    Base class for all async flow nodes.

    Design Pattern: Template Method

    Provides common structure for node execution with
    event emission and error handling.
    """

    name: str = "base_node"

    def __init__(self, ctx: RequestContext):
        self.ctx = ctx

    async def emit_event(
        self,
        event_type: EventType,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit an event to the request context."""
        event = Event(
            event_type=event_type,
            request_id=self.ctx.request_id,
            data=data or {},
        )
        await self.ctx.emit_event(event)

    async def run(self, shared: dict[str, Any]) -> str:
        """
        Execute the node with error handling.

        Args:
            shared: Shared state dictionary

        Returns:
            Next action/transition name
        """
        try:
            return await self.execute(shared)
        except Exception as e:
            logger.error(f"Node {self.name} failed: {e}", exc_info=True)
            await self.emit_event(
                EventType.ERROR,
                {"node": self.name, "error": str(e)},
            )
            return "error"

    @abstractmethod
    async def execute(self, shared: dict[str, Any]) -> str:
        """
        Execute node logic.

        Args:
            shared: Shared state dictionary

        Returns:
            Next action/transition name
        """
        pass
