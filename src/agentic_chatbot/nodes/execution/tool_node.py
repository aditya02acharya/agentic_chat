"""Execute tool node."""

from typing import Any

from ..base import AsyncBaseNode
from ...operators.registry import OperatorRegistry
from ...operators.context import OperatorContext
from ...context.assembler import ContextAssembler
from ...mcp.session import MCPSessionManager
from ...events.types import EventType
from ...utils.logging import get_logger

logger = get_logger(__name__)


class ExecuteToolNode(AsyncBaseNode):
    """
    Executes a single operator/tool.

    Handles context building and operator invocation.
    """

    name = "execute_tool"

    async def execute(self, shared: dict[str, Any]) -> str:
        decision = shared.get("decision")
        if not decision or not decision.operator:
            logger.warning("No operator specified in decision")
            return "observe"

        operator_name = decision.operator
        params = decision.params or {}

        await self.emit_event(
            EventType.TOOL_START,
            {"operator": operator_name, "params": params},
        )

        try:
            operator = OperatorRegistry.get(operator_name)
        except ValueError as e:
            logger.error(f"Unknown operator: {operator_name}")
            shared["latest_result"] = f"Error: Unknown operator {operator_name}"
            return "observe"

        assembler = ContextAssembler(
            query=shared.get("query", self.ctx.user_query),
        ).with_params(params)

        context = assembler.build(operator.context_requirements)

        mcp_session = None
        if operator.requires_mcp:
            session_manager = shared.get("mcp_session_manager")
            if session_manager:
                mcp_session = session_manager.create_session()

        try:
            result = await operator.execute(context, mcp_session)
            shared["latest_result"] = result.output if result.success else result.error

            if result.success:
                await self.emit_event(
                    EventType.TOOL_RESULT,
                    {
                        "operator": operator_name,
                        "success": True,
                        "duration_ms": result.duration_ms,
                    },
                )
            else:
                await self.emit_event(
                    EventType.TOOL_ERROR,
                    {
                        "operator": operator_name,
                        "error": result.error,
                    },
                )

        except Exception as e:
            logger.error(f"Operator {operator_name} failed: {e}", exc_info=True)
            shared["latest_result"] = f"Error: {str(e)}"
            await self.emit_event(
                EventType.TOOL_ERROR,
                {"operator": operator_name, "error": str(e)},
            )

        finally:
            if mcp_session:
                await mcp_session.close()

        return "observe"
