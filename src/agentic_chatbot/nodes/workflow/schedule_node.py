"""Schedule steps node."""

from typing import Any

from ..base import AsyncBaseNode
from ...utils.logging import get_logger

logger = get_logger(__name__)


class ScheduleStepsNode(AsyncBaseNode):
    """
    Determines execution order using topological sort.
    """

    name = "schedule_steps"

    async def execute(self, shared: dict[str, Any]) -> str:
        workflow = shared.get("workflow")
        if not workflow:
            return "error"

        step_order = self._topological_sort(workflow.steps)
        shared["step_order"] = step_order
        shared["current_step_index"] = 0
        shared["total_steps"] = len(step_order)
        shared["step_results"] = {}

        return "execute_step"

    def _topological_sort(self, steps) -> list[str]:
        """Simple topological sort of steps based on dependencies."""
        step_map = {s.id: s for s in steps}
        visited = set()
        result = []

        def visit(step_id):
            if step_id in visited:
                return
            visited.add(step_id)
            step = step_map.get(step_id)
            if step:
                for dep in step.dependencies:
                    visit(dep)
                result.append(step_id)

        for step in steps:
            visit(step.id)

        return result
