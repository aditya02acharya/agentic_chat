"""Workflow executor for DAG-based step execution."""

import asyncio
import re
import time
from typing import Any

from agentic_chatbot.core.workflow import (
    WorkflowDefinition,
    WorkflowStep,
    WorkflowStatus,
    StepResult,
    WorkflowResult,
)
from agentic_chatbot.events.emitter import EventEmitter
from agentic_chatbot.events.models import (
    WorkflowCreatedEvent,
    WorkflowStepStartEvent,
    WorkflowStepCompleteEvent,
    WorkflowCompleteEvent,
)
from agentic_chatbot.mcp.session import MCPSessionManager
from agentic_chatbot.mcp.callbacks import MCPCallbacks
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.operators.context import OperatorContext
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class WorkflowExecutor:
    """
    Executes workflow steps respecting dependencies.

    Features:
    - Parallel execution of independent steps
    - Dependency resolution via topological sort
    - Progress events per step
    - Focused context per step

    Design Pattern: Chain of Responsibility (step execution)
    """

    def __init__(
        self,
        session_manager: MCPSessionManager | None = None,
        emitter: EventEmitter | None = None,
        request_id: str | None = None,
    ):
        """
        Initialize workflow executor.

        Args:
            session_manager: MCP session manager for tool calls
            emitter: Event emitter for progress events
            request_id: Request ID for events
        """
        self._session_manager = session_manager
        self._emitter = emitter
        self._request_id = request_id
        self._results: dict[str, StepResult] = {}

    async def execute(
        self,
        workflow: WorkflowDefinition,
        initial_context: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        """
        Execute workflow with parallel optimization.

        Args:
            workflow: Workflow definition to execute
            initial_context: Initial context variables (e.g., user_query)

        Returns:
            WorkflowResult with all step outputs
        """
        self._results = {}
        context = initial_context or {}

        # Emit workflow created event
        await self._emit(
            WorkflowCreatedEvent.create(
                goal=workflow.goal,
                steps=len(workflow.steps),
                request_id=self._request_id,
            )
        )

        logger.info("Executing workflow", goal=workflow.goal, steps=len(workflow.steps))

        try:
            # Build dependency graph and get execution batches
            batches = self._get_execution_batches(workflow.steps)

            # Execute batches in order
            for batch in batches:
                if len(batch) == 1:
                    # Sequential execution
                    step = batch[0]
                    result = await self._execute_step(step, context)
                    self._results[step.id] = result
                    context[step.id] = result.output
                else:
                    # Parallel execution
                    batch_results = await asyncio.gather(
                        *[self._execute_step(step, context) for step in batch],
                        return_exceptions=True,
                    )
                    for step, result in zip(batch, batch_results):
                        if isinstance(result, Exception):
                            self._results[step.id] = StepResult(
                                step_id=step.id,
                                status=WorkflowStatus.FAILED,
                                error=str(result),
                            )
                        else:
                            self._results[step.id] = result
                            context[step.id] = result.output

            # Check for failures
            failed = [
                step_id
                for step_id, result in self._results.items()
                if result.status == WorkflowStatus.FAILED
            ]

            # Emit completion event
            await self._emit(
                WorkflowCompleteEvent.create(request_id=self._request_id)
            )

            if failed:
                return WorkflowResult(
                    goal=workflow.goal,
                    status=WorkflowStatus.FAILED,
                    steps=self._results,
                    error=f"Steps failed: {', '.join(failed)}",
                )

            return WorkflowResult(
                goal=workflow.goal,
                status=WorkflowStatus.COMPLETED,
                steps=self._results,
            )

        except Exception as e:
            logger.error("Workflow execution failed", error=str(e))
            return WorkflowResult(
                goal=workflow.goal,
                status=WorkflowStatus.FAILED,
                steps=self._results,
                error=str(e),
            )

    async def _execute_step(
        self,
        step: WorkflowStep,
        context: dict[str, Any],
    ) -> StepResult:
        """Execute a single workflow step."""
        start_time = time.time()

        # Emit step start event
        await self._emit(
            WorkflowStepStartEvent.create(
                step=self._get_step_index(step.id),
                name=step.name,
                request_id=self._request_id,
            )
        )

        logger.debug("Executing step", step_id=step.id, operator=step.operator)

        try:
            # Get operator
            operator = OperatorRegistry.create(step.operator)

            # Resolve input mapping
            resolved_inputs = self._resolve_inputs(step.input_mapping, context)

            # Build operator context
            op_context = OperatorContext(
                query=resolved_inputs.get("query", context.get("user_query", "")),
                step_results={
                    dep: self._results[dep].output
                    for dep in step.depends_on
                    if dep in self._results
                },
                extra=resolved_inputs,
            )

            # Execute with MCP session if needed
            mcp_session = None
            if operator.requires_mcp and self._session_manager:
                if operator.mcp_tools:
                    callbacks = MCPCallbacks()
                    async with self._session_manager.session_for_tool(
                        operator.mcp_tools[0], callbacks
                    ) as session:
                        result = await operator.execute(op_context, session)
                else:
                    result = await operator.execute(op_context)
            else:
                result = await operator.execute(op_context)

            duration_ms = (time.time() - start_time) * 1000

            # Emit step complete event
            await self._emit(
                WorkflowStepCompleteEvent.create(
                    step=self._get_step_index(step.id),
                    request_id=self._request_id,
                )
            )

            return StepResult(
                step_id=step.id,
                status=WorkflowStatus.COMPLETED if result.success else WorkflowStatus.FAILED,
                output=result.output,
                error=result.error,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error("Step execution failed", step_id=step.id, error=str(e))
            return StepResult(
                step_id=step.id,
                status=WorkflowStatus.FAILED,
                error=str(e),
                duration_ms=duration_ms,
            )

    def _get_execution_batches(
        self,
        steps: list[WorkflowStep],
    ) -> list[list[WorkflowStep]]:
        """
        Get execution batches using topological sort.

        Steps with no unmet dependencies can run in parallel.

        Returns:
            List of batches, where each batch contains steps that can run in parallel
        """
        # Build adjacency info
        remaining = {step.id: step for step in steps}
        completed: set[str] = set()
        batches: list[list[WorkflowStep]] = []

        while remaining:
            # Find steps with all dependencies met
            ready = []
            for step_id, step in remaining.items():
                deps_met = all(dep in completed for dep in step.depends_on)
                if deps_met:
                    ready.append(step)

            if not ready:
                # Circular dependency or missing step
                logger.error("Cannot resolve workflow dependencies")
                break

            # Add batch and mark completed
            batches.append(ready)
            for step in ready:
                completed.add(step.id)
                del remaining[step.id]

        return batches

    def _resolve_inputs(
        self,
        input_mapping: dict[str, str],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Resolve input mapping with template variables.

        Supports: {{variable}}, {{step_id.output}}, {{step_id.field}}
        """
        resolved = {}

        for key, template in input_mapping.items():
            if not isinstance(template, str):
                resolved[key] = template
                continue

            # Find all {{variable}} patterns
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
                    elif hasattr(value, part):
                        value = getattr(value, part)
                    elif hasattr(value, "output") and part == "output":
                        value = value.output
                    else:
                        value = ""
                        break

                result = result.replace(f"{{{{{match}}}}}", str(value))

            resolved[key] = result

        return resolved

    def _get_step_index(self, step_id: str) -> int:
        """Get step index for event emission."""
        try:
            # Try to extract number from step_id
            if step_id.startswith("step_"):
                return int(step_id[5:])
            return int(step_id)
        except ValueError:
            return len(self._results) + 1

    async def _emit(self, event: Any) -> None:
        """Emit event if emitter available."""
        if self._emitter:
            await self._emitter.emit(event)
