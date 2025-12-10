"""Schedule steps node for determining execution order."""

from typing import Any

from agentic_chatbot.core.workflow import WorkflowStep, WorkflowState
from agentic_chatbot.nodes.base import AsyncBaseNode
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class ScheduleStepsNode(AsyncBaseNode):
    """
    Determine execution order using topological sort.

    Type: Workflow Node

    Analyzes dependencies and creates execution batches
    where independent steps can run in parallel.
    """

    node_name = "schedule_steps"
    description = "Schedule workflow steps for execution"

    async def prep_async(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Get workflow definition."""
        workflow = shared.get("workflow", {})
        definition = workflow.get("definition")
        state = workflow.get("state")

        if not definition:
            return {"error": "No workflow definition"}

        return {
            "steps": definition.steps,
            "state": state,
        }

    async def exec_async(self, prep_res: dict[str, Any]) -> dict[str, Any]:
        """Compute execution schedule."""
        if "error" in prep_res:
            return prep_res

        steps = prep_res["steps"]

        # Build execution batches
        batches = self._topological_sort_batches(steps)

        return {
            "batches": batches,
            "total_batches": len(batches),
            "parallel_steps": sum(len(b) for b in batches if len(b) > 1),
        }

    async def post_async(
        self,
        shared: dict[str, Any],
        prep_res: dict[str, Any],
        exec_res: dict[str, Any],
    ) -> str | None:
        """Store execution schedule."""
        if "error" in exec_res:
            logger.warning("Schedule failed", error=exec_res["error"])
            return "error"

        shared["workflow"]["schedule"] = exec_res["batches"]
        shared["workflow"]["current_batch"] = 0

        logger.info(
            "Workflow scheduled",
            batches=exec_res["total_batches"],
            parallel_steps=exec_res["parallel_steps"],
        )

        return "default"

    def _topological_sort_batches(
        self,
        steps: list[WorkflowStep],
    ) -> list[list[WorkflowStep]]:
        """
        Sort steps into execution batches.

        Steps in the same batch have no dependencies on each other
        and can run in parallel.
        """
        remaining = {step.id: step for step in steps}
        completed: set[str] = set()
        batches: list[list[WorkflowStep]] = []

        while remaining:
            # Find steps with all dependencies satisfied
            ready = []
            for step_id, step in remaining.items():
                deps_met = all(dep in completed for dep in step.depends_on)
                if deps_met:
                    ready.append(step)

            if not ready:
                # Remaining steps have unmet dependencies
                logger.error("Unresolvable dependencies in workflow")
                break

            batches.append(ready)
            for step in ready:
                completed.add(step.id)
                del remaining[step.id]

        return batches
