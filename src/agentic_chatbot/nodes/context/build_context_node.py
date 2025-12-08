"""Build context node for preparing operator context."""

from typing import Any

from ..base import AsyncBaseNode
from ...context.assembler import ContextAssembler
from ...utils.logging import get_logger

logger = get_logger(__name__)


class BuildContextNode(AsyncBaseNode):
    """
    Builds focused context for operator execution.
    """

    name = "build_context"

    async def execute(self, shared: dict[str, Any]) -> str:
        decision = shared.get("decision")
        if not decision or not decision.operator:
            return "execute_tool"

        assembler = ContextAssembler(
            query=shared.get("query", self.ctx.user_query),
        )

        if decision.params:
            assembler.with_params(decision.params)

        shared["operator_context"] = assembler.build_for_operator(decision.operator)

        return "execute_tool"
