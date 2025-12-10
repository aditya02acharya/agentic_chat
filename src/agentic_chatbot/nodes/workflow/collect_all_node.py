"""Collect all results node for gathering workflow outputs."""

from typing import Any

from agentic_chatbot.core.workflow import WorkflowStatus, WorkflowResult
from agentic_chatbot.events.models import WorkflowCompleteEvent
from agentic_chatbot.nodes.base import AsyncBaseNode
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class CollectAllResultsNode(AsyncBaseNode):
    """
    Gather all step outputs into workflow result.

    Type: Workflow Node

    Collects results from all completed steps and
    builds the final WorkflowResult.
    """

    node_name = "collect_all_results"
    description = "Collect all workflow step results"

    async def prep_async(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Get workflow state with results."""
        workflow = shared.get("workflow", {})
        definition = workflow.get("definition")
        state = workflow.get("state")

        if not definition or not state:
            return {"error": "No workflow state"}

        return {
            "goal": definition.goal,
            "step_results": state.step_results,
            "expected_steps": [s.id for s in definition.steps],
        }

    async def exec_async(self, prep_res: dict[str, Any]) -> dict[str, Any]:
        """Build workflow result."""
        if "error" in prep_res:
            return prep_res

        goal = prep_res["goal"]
        step_results = prep_res["step_results"]
        expected_steps = prep_res["expected_steps"]

        # Check completion
        completed_steps = set(step_results.keys())
        missing_steps = set(expected_steps) - completed_steps

        # Check for failures
        failed_steps = [
            step_id
            for step_id, result in step_results.items()
            if result.status == WorkflowStatus.FAILED
        ]

        # Determine overall status
        if missing_steps:
            status = WorkflowStatus.FAILED
            error = f"Missing steps: {', '.join(missing_steps)}"
        elif failed_steps:
            status = WorkflowStatus.FAILED
            error = f"Failed steps: {', '.join(failed_steps)}"
        else:
            status = WorkflowStatus.COMPLETED
            error = None

        result = WorkflowResult(
            goal=goal,
            status=status,
            steps=step_results,
            error=error,
        )

        return {"result": result}

    async def post_async(
        self,
        shared: dict[str, Any],
        prep_res: dict[str, Any],
        exec_res: dict[str, Any],
    ) -> str | None:
        """Store workflow result."""
        if "error" in exec_res:
            logger.warning("Workflow collection failed", error=exec_res["error"])
            return "error"

        result: WorkflowResult = exec_res["result"]

        # Store in results
        shared.setdefault("results", {})
        shared["results"]["workflow_output"] = result

        # Emit completion event
        await self.emit_event(
            shared,
            WorkflowCompleteEvent.create(request_id=shared.get("request_id")),
        )

        logger.info(
            "Workflow complete",
            status=result.status.value,
            steps=len(result.steps),
            failed=len(result.failed_steps),
        )

        return "default"
