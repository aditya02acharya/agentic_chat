"""Handle blocked node for graceful degradation."""

from typing import Any

from ..base import AsyncBaseNode
from ...events.types import EventType
from ...utils.logging import get_logger

logger = get_logger(__name__)


class HandleBlockedNode(AsyncBaseNode):
    """
    Handles cases where the system is stuck.

    Provides graceful degradation and user communication.
    """

    name = "handle_blocked"

    async def execute(self, shared: dict[str, Any]) -> str:
        reflection = shared.get("reflection")
        previous_results = shared.get("previous_results", [])

        await self.emit_event(
            EventType.THINKING_UPDATE,
            {"phase": "blocked", "reason": "Unable to proceed"},
        )

        if previous_results:
            shared["final_response"] = (
                "I encountered some difficulties, but here's what I found:\n\n"
                + "\n\n".join(str(r)[:500] for r in previous_results)
            )
        else:
            shared["final_response"] = (
                "I apologize, but I was unable to complete your request. "
                "Could you please try rephrasing your question or providing more details?"
            )

        return "write"
