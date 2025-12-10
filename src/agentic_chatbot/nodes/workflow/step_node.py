"""Execute step node for running single workflow step."""

import time
from typing import Any

from agentic_chatbot.core.workflow import WorkflowStep, WorkflowStatus, StepResult
from agentic_chatbot.events.models import WorkflowStepStartEvent, WorkflowStepCompleteEvent
from agentic_chatbot.mcp.callbacks import MCPCallbacks
from agentic_chatbot.nodes.base import AsyncBaseNode
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.operators.context import OperatorContext
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class ExecuteStepNode(AsyncBaseNode):
    """
    Execute a single workflow step.

    Type: Workflow Node

    Runs the operator for a specific step with
    resolved inputs from previous steps.
    """

    node_name = "execute_step"
    description = "Execute single workflow step"

    async def prep_async(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Get current step to execute."""
        workflow = shared.get("workflow", {})
        schedule = workflow.get("schedule", [])
        current_batch = workflow.get("current_batch", 0)
        step_index = workflow.get("current_step_in_batch", 0)

        if current_batch >= len(schedule):
            return {"error": "No more batches"}

        batch = schedule[current_batch]
        if step_index >= len(batch):
            return {"error": "No more steps in batch"}

        step = batch[step_index]

        # Get results from previous steps
        state = workflow.get("state")
        step_results = state.step_results if state else {}

        return {
            "step": step,
            "step_results": step_results,
            "query": shared.get("user_query", ""),
        }

    async def exec_async(self, prep_res: dict[str, Any]) -> dict[str, Any]:
        """Execute the step."""
        if "error" in prep_res:
            return prep_res

        step: WorkflowStep = prep_res["step"]
        step_results = prep_res["step_results"]
        query = prep_res["query"]

        start_time = time.time()

        try:
            # Get operator
            operator = OperatorRegistry.create(step.operator)

            # Resolve input mapping
            resolved_inputs = self._resolve_inputs(
                step.input_mapping,
                step_results,
                query,
            )

            # Build context
            context = OperatorContext(
                query=resolved_inputs.get("query", query),
                step_results={
                    dep: step_results[dep].output
                    for dep in step.depends_on
                    if dep in step_results
                },
                extra=resolved_inputs,
            )

            # Execute operator
            result = await operator.execute(context)

            duration_ms = (time.time() - start_time) * 1000

            return {
                "step_id": step.id,
                "result": StepResult(
                    step_id=step.id,
                    status=WorkflowStatus.COMPLETED if result.success else WorkflowStatus.FAILED,
                    output=result.output,
                    error=result.error,
                    duration_ms=duration_ms,
                ),
            }

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error("Step execution failed", step_id=step.id, error=str(e))
            return {
                "step_id": step.id,
                "result": StepResult(
                    step_id=step.id,
                    status=WorkflowStatus.FAILED,
                    error=str(e),
                    duration_ms=duration_ms,
                ),
            }

    async def post_async(
        self,
        shared: dict[str, Any],
        prep_res: dict[str, Any],
        exec_res: dict[str, Any],
    ) -> str | None:
        """Store step result and emit events."""
        if "error" in exec_res:
            return "error"

        step = prep_res["step"]
        result: StepResult = exec_res["result"]

        # Emit step start event (before) and complete event (after)
        await self.emit_event(
            shared,
            WorkflowStepCompleteEvent.create(
                step=self._get_step_number(step.id),
                request_id=shared.get("request_id"),
            ),
        )

        # Store result in workflow state
        state = shared["workflow"].get("state")
        if state:
            state.step_results[step.id] = result

        # Move to next step in batch
        shared["workflow"]["current_step_in_batch"] = (
            shared["workflow"].get("current_step_in_batch", 0) + 1
        )

        logger.debug(
            "Step executed",
            step_id=step.id,
            status=result.status.value,
            duration_ms=result.duration_ms,
        )

        return "default"

    def _resolve_inputs(
        self,
        input_mapping: dict[str, str],
        step_results: dict[str, StepResult],
        query: str,
    ) -> dict[str, Any]:
        """Resolve input mapping templates."""
        import re

        resolved = {}
        context = {"user_query": query}

        # Add step outputs to context
        for step_id, result in step_results.items():
            context[step_id] = result.output

        for key, template in input_mapping.items():
            if not isinstance(template, str):
                resolved[key] = template
                continue

            # Find {{variable}} patterns
            pattern = r"\{\{([^}]+)\}\}"
            matches = re.findall(pattern, template)

            if not matches:
                resolved[key] = template
                continue

            result = template
            for match in matches:
                parts = match.split(".")
                value = context

                for part in parts:
                    if isinstance(value, dict):
                        value = value.get(part, "")
                    else:
                        value = ""
                        break

                result = result.replace(f"{{{{{match}}}}}", str(value))

            resolved[key] = result

        return resolved

    def _get_step_number(self, step_id: str) -> int:
        """Extract step number from ID."""
        try:
            if step_id.startswith("step_"):
                return int(step_id[5:])
            return int(step_id)
        except ValueError:
            return 1
