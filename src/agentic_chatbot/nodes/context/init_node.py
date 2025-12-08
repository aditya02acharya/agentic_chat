"""Initialize node for setting up request context."""

from typing import Any

from ..base import AsyncBaseNode
from ...events.types import EventType
from ...utils.logging import get_logger

logger = get_logger(__name__)


class InitializeNode(AsyncBaseNode):
    """
    Initializes the shared state for a chat request.
    """

    name = "initialize"

    async def execute(self, shared: dict[str, Any]) -> str:
        shared["query"] = self.ctx.user_query
        shared["conversation_id"] = self.ctx.conversation_id
        shared["request_id"] = self.ctx.request_id
        shared["iteration"] = 0
        shared["max_iterations"] = 5
        shared["previous_results"] = []
        shared["action_history"] = ""
        shared["decision"] = None
        shared["reflection"] = None
        shared["synthesized_content"] = None
        shared["final_response"] = None

        await self.emit_event(
            EventType.THINKING_START,
            {"phase": "initialization", "query": self.ctx.user_query[:100]},
        )

        logger.info(
            f"Initialized request {self.ctx.request_id}",
            query=self.ctx.user_query[:50],
        )

        return "fetch_tools"
