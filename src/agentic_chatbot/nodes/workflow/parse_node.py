"""Parse workflow node for building DAG from definition."""

from typing import Any

from agentic_chatbot.core.workflow import (
    WorkflowDefinition,
    WorkflowStep,
    WorkflowState,
    WorkflowDefinitionSchema,
)
from agentic_chatbot.events.models import WorkflowCreatedEvent
from agentic_chatbot.nodes.base import AsyncBaseNode
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class ParseWorkflowNode(AsyncBaseNode):
    """
    Convert workflow plan to executable DAG.

    Type: Workflow Node

    Parses the workflow definition from the supervisor's
    decision and prepares it for execution.
    """

    node_name = "parse_workflow"
    description = "Parse workflow definition into executable DAG"

    async def prep_async(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Get workflow definition from supervisor decision."""
        decision = shared.get("supervisor", {}).get("current_decision")

        if not decision or decision.action != "CREATE_WORKFLOW":
            return {"error": "No workflow in decision"}

        return {
            "goal": decision.goal,
            "steps": decision.steps,
            "query": shared.get("user_query", ""),
        }

    async def exec_async(self, prep_res: dict[str, Any]) -> dict[str, Any]:
        """Parse and validate workflow definition."""
        if "error" in prep_res:
            return prep_res

        goal = prep_res.get("goal", "")
        steps_data = prep_res.get("steps", [])

        if not steps_data:
            return {"error": "No steps in workflow"}

        try:
            # Convert to WorkflowStep objects
            steps = []
            for step_data in steps_data:
                step = WorkflowStep(
                    id=step_data.get("id", f"step_{len(steps)}"),
                    name=step_data.get("name", "Unnamed step"),
                    operator=step_data.get("operator", ""),
                    input_mapping=step_data.get("input_mapping", {}),
                    depends_on=step_data.get("depends_on", []),
                )
                steps.append(step)

            # Validate dependencies
            step_ids = {s.id for s in steps}
            for step in steps:
                for dep in step.depends_on:
                    if dep not in step_ids:
                        return {"error": f"Unknown dependency: {dep}"}

            # Check for cycles (simple check)
            if self._has_cycle(steps):
                return {"error": "Workflow has circular dependencies"}

            workflow = WorkflowDefinition(goal=goal, steps=steps)

            return {
                "workflow": workflow,
                "step_count": len(steps),
            }

        except Exception as e:
            logger.error("Failed to parse workflow", error=str(e))
            return {"error": f"Parse error: {str(e)}"}

    async def post_async(
        self,
        shared: dict[str, Any],
        prep_res: dict[str, Any],
        exec_res: dict[str, Any],
    ) -> str | None:
        """Store parsed workflow."""
        if "error" in exec_res:
            logger.warning("Workflow parse failed", error=exec_res["error"])
            shared["workflow_error"] = exec_res["error"]
            return "error"

        workflow = exec_res["workflow"]

        # Initialize workflow state
        shared.setdefault("workflow", {})
        shared["workflow"]["definition"] = workflow
        shared["workflow"]["state"] = WorkflowState(definition=workflow)

        # Emit workflow created event
        await self.emit_event(
            shared,
            WorkflowCreatedEvent.create(
                goal=workflow.goal,
                steps=len(workflow.steps),
                request_id=shared.get("request_id"),
            ),
        )

        logger.info(
            "Workflow parsed",
            goal=workflow.goal,
            steps=exec_res["step_count"],
        )

        return "default"

    def _has_cycle(self, steps: list[WorkflowStep]) -> bool:
        """Check for circular dependencies using DFS."""
        step_map = {s.id: s for s in steps}
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(step_id: str) -> bool:
            visited.add(step_id)
            rec_stack.add(step_id)

            step = step_map.get(step_id)
            if step:
                for dep in step.depends_on:
                    if dep not in visited:
                        if dfs(dep):
                            return True
                    elif dep in rec_stack:
                        return True

            rec_stack.remove(step_id)
            return False

        for step in steps:
            if step.id not in visited:
                if dfs(step.id):
                    return True

        return False
