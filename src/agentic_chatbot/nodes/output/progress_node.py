"""Emit progress node for sending progress events."""

from typing import Any

from agentic_chatbot.events.models import Event, ToolProgressEvent
from agentic_chatbot.events.types import EventType
from agentic_chatbot.nodes.base import AsyncBaseNode
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class EmitProgressNode(AsyncBaseNode):
    """
    Send progress event to user.

    Type: Output Node

    Emits a progress update to keep the user informed
    about what's happening.
    """

    node_name = "emit_progress"
    description = "Emit progress event"

    def __init__(self, message: str = "", progress: float = 0.0, **kwargs: Any):
        """
        Initialize with progress info.

        Args:
            message: Progress message to display
            progress: Progress value (0.0 to 1.0)
        """
        super().__init__(**kwargs)
        self._default_message = message
        self._default_progress = progress

    async def prep_async(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Get progress information."""
        # Check if there's dynamic progress info
        current_progress = shared.get("current_progress", {})

        return {
            "message": current_progress.get("message", self._default_message),
            "progress": current_progress.get("value", self._default_progress),
            "tool": current_progress.get("tool", "system"),
            "request_id": shared.get("request_id"),
        }

    async def exec_async(self, prep_res: dict[str, Any]) -> Event:
        """Create progress event."""
        return ToolProgressEvent.create(
            tool=prep_res["tool"],
            progress=prep_res["progress"],
            message=prep_res["message"],
            request_id=prep_res["request_id"],
        )

    async def post_async(
        self,
        shared: dict[str, Any],
        prep_res: dict[str, Any],
        exec_res: Event,
    ) -> str | None:
        """Emit progress event."""
        await self.emit_event(shared, exec_res)

        logger.debug(
            "Progress emitted",
            tool=prep_res["tool"],
            progress=prep_res["progress"],
        )

        return "default"
