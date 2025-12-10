"""Stream node for SSE chunk streaming."""

from typing import Any

from agentic_chatbot.events.models import ResponseChunkEvent, ResponseDoneEvent
from agentic_chatbot.nodes.base import AsyncBaseNode
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class StreamNode(AsyncBaseNode):
    """
    Stream response chunks via SSE.

    Type: Output Node

    Splits the final response into chunks and
    emits them as SSE events.
    """

    node_name = "stream"
    description = "Stream response chunks"

    # Chunk size for streaming
    CHUNK_SIZE = 50  # characters per chunk

    async def prep_async(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Get response to stream."""
        results = shared.get("results", {})
        response = results.get("final_response", "")

        return {
            "response": response,
            "request_id": shared.get("request_id"),
        }

    async def exec_async(self, prep_res: dict[str, Any]) -> list[str]:
        """Split response into chunks."""
        response = prep_res.get("response", "")

        if not response:
            return []

        # Split into chunks
        chunks = []
        for i in range(0, len(response), self.CHUNK_SIZE):
            chunk = response[i : i + self.CHUNK_SIZE]
            chunks.append(chunk)

        return chunks

    async def post_async(
        self,
        shared: dict[str, Any],
        prep_res: dict[str, Any],
        exec_res: list[str],
    ) -> str | None:
        """Stream chunks as events."""
        request_id = prep_res.get("request_id")

        # Emit each chunk
        for chunk in exec_res:
            await self.emit_event(
                shared,
                ResponseChunkEvent.create(
                    content=chunk,
                    request_id=request_id,
                ),
            )

        # Emit done event
        await self.emit_event(
            shared,
            ResponseDoneEvent.create(request_id=request_id),
        )

        logger.info(
            "Response streamed",
            chunks=len(exec_res),
            total_length=sum(len(c) for c in exec_res),
        )

        return "default"
