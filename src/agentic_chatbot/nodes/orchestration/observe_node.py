"""Observe node for collecting and processing results."""

from typing import Any

from ..base import AsyncBaseNode
from ...events.types import EventType
from ...utils.logging import get_logger

logger = get_logger(__name__)


class ObserveNode(AsyncBaseNode):
    """
    Collects and compresses results from tool/workflow execution.

    Prepares results for the reflection phase.
    """

    name = "observe"

    async def execute(self, shared: dict[str, Any]) -> str:
        latest_result = shared.get("latest_result")
        if not latest_result:
            return "reflect"

        previous_results = shared.get("previous_results", [])
        previous_results.append(latest_result)
        shared["previous_results"] = previous_results

        action_history = shared.get("action_history", "")
        decision = shared.get("decision")
        if decision:
            action_entry = f"- {decision.action}"
            if decision.operator:
                action_entry += f" ({decision.operator})"
            result_summary = str(latest_result)[:200] if latest_result else "No result"
            action_entry += f": {result_summary}"
            action_history += action_entry + "\n"
            shared["action_history"] = action_history

        await self.emit_event(
            EventType.TOOL_RESULT,
            {
                "result_count": len(previous_results),
                "latest_summary": str(latest_result)[:200] if latest_result else None,
            },
        )

        return "reflect"
