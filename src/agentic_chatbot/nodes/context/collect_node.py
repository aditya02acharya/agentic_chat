"""Collect result node for storing operator output."""

from typing import Any

from agentic_chatbot.nodes.base import AsyncBaseNode
from agentic_chatbot.operators.context import OperatorResult
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class CollectResultNode(AsyncBaseNode):
    """
    Store operator output in result store.

    Type: Context Node

    Takes the output from an operator execution
    and stores it properly for later use.
    """

    node_name = "collect_result"
    description = "Collect and store operator result"

    async def prep_async(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Get result to collect."""
        # Get the most recent tool output
        results = shared.get("results", {})
        tool_outputs = results.get("tool_outputs", [])

        latest_result = tool_outputs[-1] if tool_outputs else None

        # Get result store
        result_store = shared.get("result_store")

        return {
            "result": latest_result,
            "result_store": result_store,
            "output_index": len(tool_outputs),
        }

    async def exec_async(self, prep_res: dict[str, Any]) -> dict[str, Any]:
        """Process result for collection."""
        result = prep_res.get("result")

        if result is None:
            return {"stored": False, "reason": "No result to collect"}

        # Extract key information
        if isinstance(result, OperatorResult):
            return {
                "stored": True,
                "output": result.output,
                "success": result.success,
                "has_contents": result.has_contents,
                "content_count": len(result.contents),
            }
        else:
            return {
                "stored": True,
                "output": result,
                "success": True,
                "has_contents": False,
                "content_count": 0,
            }

    async def post_async(
        self,
        shared: dict[str, Any],
        prep_res: dict[str, Any],
        exec_res: dict[str, Any],
    ) -> str | None:
        """Store in result store."""
        result_store = prep_res.get("result_store")
        result = prep_res.get("result")

        if result_store and result and exec_res.get("stored"):
            result_store.store_tool_output(result)

            logger.debug(
                "Result collected",
                index=prep_res["output_index"],
                success=exec_res["success"],
                has_contents=exec_res["has_contents"],
            )
        elif not exec_res.get("stored"):
            logger.debug("No result to collect", reason=exec_res.get("reason"))

        return "default"
