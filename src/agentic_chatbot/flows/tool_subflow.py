"""Tool subflow for executing single tools."""

from typing import Any

from ..core.request_context import RequestContext
from ..nodes.context.build_context_node import BuildContextNode
from ..nodes.execution.tool_node import ExecuteToolNode
from ..nodes.context.collect_node import CollectResultNode
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ToolSubFlow:
    """
    Subflow for executing a single tool/operator.

    Handles context building, execution, and result collection.
    """

    def __init__(self, ctx: RequestContext):
        self.ctx = ctx
        self._nodes = {
            "build_context": BuildContextNode(ctx),
            "execute_tool": ExecuteToolNode(ctx),
            "collect": CollectResultNode(ctx),
        }

    async def run(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool subflow."""
        await self._nodes["build_context"].run(shared)
        await self._nodes["execute_tool"].run(shared)
        await self._nodes["collect"].run(shared)

        return shared
