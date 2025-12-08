"""Stream node for SSE output."""

from typing import Any

from ..base import AsyncBaseNode
from ...events.models import ContentEvent, Event
from ...events.types import EventType
from ...utils.logging import get_logger

logger = get_logger(__name__)


class StreamNode(AsyncBaseNode):
    """
    Streams the final response via SSE.
    """

    name = "stream"

    async def execute(self, shared: dict[str, Any]) -> str:
        response = shared.get("final_response", "")

        await self.emit_event(EventType.RESPONSE_START, {})

        chunk_size = 50
        for i in range(0, len(response), chunk_size):
            chunk = response[i : i + chunk_size]
            await self.ctx.emit_event(
                ContentEvent(
                    event_type=EventType.RESPONSE_CHUNK,
                    request_id=self.ctx.request_id,
                    content=chunk,
                    content_type="text/plain",
                )
            )

        await self.emit_event(
            EventType.RESPONSE_DONE,
            {"total_length": len(response)},
        )

        return "done"
