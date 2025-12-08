"""Execute step node."""

import time
from typing import Any

from ..base import AsyncBaseNode
from ...core.workflow import StepResult, StepStatus
from ...operators.registry import OperatorRegistry
from ...operators.context import OperatorContext
from ...events.types import EventType
from ...utils.logging import get_logger

logger = get_logger(__name__)


class ExecuteStepNode(AsyncBaseNode):
    """
    Executes a single workflow step.
    """

    name = "execute_step"

    async def execute(self, shared: dict[str, Any]) -> str:
        workflow = shared.get("workflow")
        step_order = shared.get("step_order", [])
        current_index = shared.get("current_step_index", 0)

        if current_index >= len(step_order):
            return "collect_all"

        if not workflow:
            return "collect_all"

        step_id = step_order[current_index]
        step = next((s for s in workflow.steps if s.id == step_id), None)

        if not step:
            shared["current_step_index"] = current_index + 1
            return "execute_step"

        await self.emit_event(
            EventType.WORKFLOW_STEP_START,
            {"step_id": step_id, "step_name": step.name, "operator": step.operator},
        )

        start_time = time.time()

        try:
            operator = OperatorRegistry.get(step.operator)
            step_query = step.description or (workflow.goal if workflow else "")
            context = OperatorContext(
                query=step_query,
                params=step.params,
                previous_results=list(shared.get("step_results", {}).values()),
            )

            mcp_session = None
            if operator.requires_mcp:
                session_manager = shared.get("mcp_session_manager")
                if session_manager:
                    mcp_session = session_manager.create_session()

            try:
                result = await operator.execute(context, mcp_session)
                duration_ms = (time.time() - start_time) * 1000

                step_result = StepResult(
                    step_id=step_id,
                    status=StepStatus.COMPLETED if result.success else StepStatus.FAILED,
                    output=result.output,
                    error=result.error,
                    duration_ms=duration_ms,
                )
            finally:
                if mcp_session:
                    await mcp_session.close()

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            step_result = StepResult(
                step_id=step_id,
                status=StepStatus.FAILED,
                error=str(e),
                duration_ms=duration_ms,
            )

        shared["step_results"][step_id] = step_result

        await self.emit_event(
            EventType.WORKFLOW_STEP_COMPLETE,
            {
                "step_id": step_id,
                "status": step_result.status.value,
                "duration_ms": step_result.duration_ms,
            },
        )

        shared["current_step_index"] = current_index + 1
        shared["current_step_name"] = step.name

        return "execute_step"
