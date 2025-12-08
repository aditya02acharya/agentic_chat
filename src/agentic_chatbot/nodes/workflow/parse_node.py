"""Parse workflow node."""

from typing import Any

from ..base import AsyncBaseNode
from ...core.workflow import WorkflowDefinition, WorkflowStep
from ...events.types import EventType
from ...utils.logging import get_logger

logger = get_logger(__name__)


class ParseWorkflowNode(AsyncBaseNode):
    """
    Parses workflow definition from supervisor decision.
    """

    name = "parse_workflow"

    async def execute(self, shared: dict[str, Any]) -> str:
        decision = shared.get("decision")
        if not decision or decision.action != "CREATE_WORKFLOW":
            return "error"

        steps = []
        for i, step_data in enumerate(decision.steps or []):
            step = WorkflowStep(
                id=step_data.get("id", f"step_{i}"),
                name=step_data.get("name", f"Step {i + 1}"),
                operator=step_data.get("operator", ""),
                params=step_data.get("params", {}),
                dependencies=step_data.get("dependencies", []),
                description=step_data.get("description"),
            )
            steps.append(step)

        workflow = WorkflowDefinition(
            goal=decision.goal or "Complete the workflow",
            steps=steps,
        )

        shared["workflow"] = workflow
        shared["workflow_status"] = "pending"

        await self.emit_event(
            EventType.WORKFLOW_START,
            {"goal": workflow.goal, "step_count": len(steps)},
        )

        return "schedule"
