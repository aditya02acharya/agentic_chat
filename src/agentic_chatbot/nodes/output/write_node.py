"""Write node for formatting final response."""

from typing import Any

from ..base import AsyncBaseNode
from ...operators.registry import OperatorRegistry
from ...operators.context import OperatorContext
from ...events.types import EventType
from ...utils.logging import get_logger

logger = get_logger(__name__)


class WriteNode(AsyncBaseNode):
    """
    Formats the final response for the user.
    """

    name = "write"

    async def execute(self, shared: dict[str, Any]) -> str:
        if shared.get("final_response"):
            return "stream"

        decision = shared.get("decision")
        if decision and decision.action == "ANSWER" and decision.response:
            shared["final_response"] = decision.response
            return "stream"

        content = shared.get("synthesized_content")
        if not content:
            previous_results = shared.get("previous_results", [])
            if previous_results:
                content = "\n\n".join(str(r) for r in previous_results)
            else:
                content = "I'm sorry, I couldn't find relevant information for your query."

        try:
            writer = OperatorRegistry.get("writer")
            context = OperatorContext(
                query=shared.get("query", self.ctx.user_query),
                previous_results=[content],
            )
            context.params["content"] = content

            result = await writer.execute(context)

            if result.success:
                shared["final_response"] = result.output
            else:
                shared["final_response"] = content

        except Exception as e:
            logger.error(f"Writer failed: {e}", exc_info=True)
            shared["final_response"] = content

        return "stream"
