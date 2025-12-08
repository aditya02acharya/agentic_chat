"""Collect all results node."""

from typing import Any

from ..base import AsyncBaseNode
from ...core.workflow import WorkflowResult
from ...events.types import EventType
from ...utils.logging import get_logger

logger = get_logger(__name__)


class CollectAllResultsNode(AsyncBaseNode):
    """
    Collects all workflow step results.
    """

    name = "collect_all"

    async def execute(self, shared: dict[str, Any]) -> str:
        workflow = shared.get("workflow")
        step_results = shared.get("step_results", {})

        if workflow:
            workflow_result = WorkflowResult(
                definition=workflow,
                step_results=step_results,
                status="completed",
            )
            shared["workflow_result"] = workflow_result

            outputs = []
            for result in step_results.values():
                if result.output:
                    outputs.append(str(result.output))

            shared["previous_results"] = outputs

        await self.emit_event(
            EventType.WORKFLOW_COMPLETE,
            {"step_count": len(step_results)},
        )

        return "observe"
