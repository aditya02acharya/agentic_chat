"""Workflow subflow for multi-step execution."""

from typing import Any

from ..core.request_context import RequestContext
from ..nodes.workflow.parse_node import ParseWorkflowNode
from ..nodes.workflow.schedule_node import ScheduleStepsNode
from ..nodes.workflow.step_node import ExecuteStepNode
from ..nodes.workflow.collect_all_node import CollectAllResultsNode
from ..utils.logging import get_logger

logger = get_logger(__name__)


class WorkflowSubFlow:
    """
    Subflow for executing multi-step workflows.

    Handles workflow parsing, scheduling, and step execution.
    """

    def __init__(self, ctx: RequestContext):
        self.ctx = ctx
        self._nodes = {
            "parse": ParseWorkflowNode(ctx),
            "schedule": ScheduleStepsNode(ctx),
            "execute_step": ExecuteStepNode(ctx),
            "collect_all": CollectAllResultsNode(ctx),
        }

    async def run(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Execute the workflow subflow."""
        await self._nodes["parse"].run(shared)
        await self._nodes["schedule"].run(shared)

        step_order = shared.get("step_order", [])
        for _ in step_order:
            await self._nodes["execute_step"].run(shared)

        await self._nodes["collect_all"].run(shared)

        return shared
