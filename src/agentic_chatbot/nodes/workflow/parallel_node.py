"""Execute parallel node for running multiple steps concurrently."""

import asyncio
import time
from typing import Any

from agentic_chatbot.core.workflow import WorkflowStep, WorkflowStatus, StepResult
from agentic_chatbot.events.models import WorkflowStepStartEvent, WorkflowStepCompleteEvent
from agentic_chatbot.nodes.base import AsyncBaseNode
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.operators.context import OperatorContext
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class ExecuteParallelNode(AsyncBaseNode):
    """
    Execute multiple workflow steps concurrently.

    Type: Workflow Node (uses asyncio.gather)

    Runs independent steps in parallel for better performance.
    """

    node_name = "execute_parallel"
    description = "Execute workflow steps in parallel"

    async def prep_async(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Get steps to execute in parallel."""
        workflow = shared.get("workflow", {})
        schedule = workflow.get("schedule", [])
        current_batch = workflow.get("current_batch", 0)

        if current_batch >= len(schedule):
            return {"error": "No more batches"}

        batch = schedule[current_batch]
        if len(batch) <= 1:
            return {"error": "Use ExecuteStepNode for single steps"}

        # Get results from previous steps
        state = workflow.get("state")
        step_results = state.step_results if state else {}

        return {
            "steps": batch,
            "step_results": step_results,
            "query": shared.get("user_query", ""),
        }

    async def exec_async(self, prep_res: dict[str, Any]) -> dict[str, Any]:
        """Execute steps in parallel."""
        if "error" in prep_res:
            return prep_res

        steps = prep_res["steps"]
        step_results = prep_res["step_results"]
        query = prep_res["query"]

        # Create tasks for parallel execution
        tasks = [
            self._execute_single_step(step, step_results, query)
            for step in steps
        ]

        # Run all in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        step_result_map = {}
        for step, result in zip(steps, results):
            if isinstance(result, Exception):
                step_result_map[step.id] = StepResult(
                    step_id=step.id,
                    status=WorkflowStatus.FAILED,
                    error=str(result),
                )
            else:
                step_result_map[step.id] = result

        return {"results": step_result_map}

    async def post_async(
        self,
        shared: dict[str, Any],
        prep_res: dict[str, Any],
        exec_res: dict[str, Any],
    ) -> str | None:
        """Store all step results."""
        if "error" in exec_res:
            return "error"

        results = exec_res["results"]

        # Store results in workflow state
        state = shared["workflow"].get("state")
        if state:
            state.step_results.update(results)

        # Emit completion events for each step
        for step_id, result in results.items():
            await self.emit_event(
                shared,
                WorkflowStepCompleteEvent.create(
                    step=self._get_step_number(step_id),
                    request_id=shared.get("request_id"),
                ),
            )

        # Move to next batch
        shared["workflow"]["current_batch"] = (
            shared["workflow"].get("current_batch", 0) + 1
        )
        shared["workflow"]["current_step_in_batch"] = 0

        logger.info(
            "Parallel execution complete",
            steps=len(results),
            failed=sum(1 for r in results.values() if r.status == WorkflowStatus.FAILED),
        )

        return "default"

    async def _execute_single_step(
        self,
        step: WorkflowStep,
        step_results: dict[str, StepResult],
        query: str,
    ) -> StepResult:
        """Execute a single step (for parallel execution)."""
        import re

        start_time = time.time()

        try:
            operator = OperatorRegistry.create(step.operator)

            # Resolve inputs
            resolved_inputs = {}
            for key, template in step.input_mapping.items():
                if not isinstance(template, str):
                    resolved_inputs[key] = template
                    continue

                pattern = r"\{\{([^}]+)\}\}"
                matches = re.findall(pattern, template)

                if not matches:
                    resolved_inputs[key] = template
                    continue

                result = template
                context = {"user_query": query}
                for sid, sr in step_results.items():
                    context[sid] = sr.output

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

                resolved_inputs[key] = result

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

            # Execute
            op_result = await operator.execute(context)

            duration_ms = (time.time() - start_time) * 1000

            return StepResult(
                step_id=step.id,
                status=WorkflowStatus.COMPLETED if op_result.success else WorkflowStatus.FAILED,
                output=op_result.output,
                error=op_result.error,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return StepResult(
                step_id=step.id,
                status=WorkflowStatus.FAILED,
                error=str(e),
                duration_ms=duration_ms,
            )

    def _get_step_number(self, step_id: str) -> int:
        """Extract step number from ID."""
        try:
            if step_id.startswith("step_"):
                return int(step_id[5:])
            return int(step_id)
        except ValueError:
            return 1
