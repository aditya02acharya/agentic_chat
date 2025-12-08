"""Progress emission node."""

from typing import Any

from ..base import AsyncBaseNode
from ...events.models import ProgressEvent
from ...events.types import EventType
from ...utils.logging import get_logger

logger = get_logger(__name__)


class EmitProgressNode(AsyncBaseNode):
    """
    Emits progress updates during workflow execution.
    """

    name = "emit_progress"

    async def execute(self, shared: dict[str, Any]) -> str:
        current_step = shared.get("current_step_index", 0)
        total_steps = shared.get("total_steps", 1)
        step_name = shared.get("current_step_name", "Processing")

        progress = current_step / max(total_steps, 1)

        await self.ctx.emit_event(
            ProgressEvent(
                event_type=EventType.WORKFLOW_STEP_COMPLETE,
                request_id=self.ctx.request_id,
                progress=progress,
                message=f"Completed: {step_name}",
            )
        )

        return "continue"
