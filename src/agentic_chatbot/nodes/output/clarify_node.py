"""Clarify node for requesting user input."""

from typing import Any

from ..base import AsyncBaseNode
from ...events.types import EventType
from ...utils.logging import get_logger

logger = get_logger(__name__)


class ClarifyNode(AsyncBaseNode):
    """
    Requests clarification from the user.
    """

    name = "clarify"

    async def execute(self, shared: dict[str, Any]) -> str:
        decision = shared.get("decision")
        question = "Could you please provide more details?"

        if decision and decision.question:
            question = decision.question

        await self.emit_event(
            EventType.CLARIFY_REQUEST,
            {"question": question},
        )

        shared["final_response"] = question

        return "stream"
