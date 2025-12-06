"""Execute tool node for running single operators."""

from typing import Any

from agentic_chatbot.core.exceptions import OperatorError
from agentic_chatbot.events.models import ToolStartEvent, ToolCompleteEvent, ToolErrorEvent
from agentic_chatbot.mcp.callbacks import MCPCallbacks
from agentic_chatbot.nodes.base import AsyncBaseNode
from agentic_chatbot.operators.base import OperatorType
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.operators.context import OperatorContext, OperatorResult
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class ExecuteToolNode(AsyncBaseNode):
    """
    Execute single operator with focused context.

    Type: Execution Node (with retry via PocketFlow)

    Runs the operator specified in the supervisor's decision,
    handling both pure LLM and MCP-backed operators.
    """

    node_name = "execute_tool"
    description = "Execute a single operator"

    def __init__(self, max_retries: int = 2, **kwargs: Any):
        """Initialize with retry config."""
        super().__init__(max_retries=max_retries, **kwargs)

    async def prep_async(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Prepare operator context."""
        decision = shared.get("supervisor", {}).get("current_decision")
        if not decision:
            raise OperatorError("No decision available")

        operator_name = decision.operator
        if not operator_name:
            raise OperatorError("No operator specified in decision")

        # Get operator
        try:
            operator = OperatorRegistry.create(operator_name)
        except KeyError:
            raise OperatorError(f"Unknown operator: {operator_name}")

        # Build context for operator
        context = OperatorContext(
            query=shared.get("user_query", ""),
            recent_messages=shared.get("recent_messages", []),
            conversation_summary=shared.get("memory", {}).get("summary", "")
            if isinstance(shared.get("memory"), dict)
            else "",
            shared_store=shared,
        )

        # Add params from decision
        if decision.params:
            context.extra.update(decision.params)

        # Emit start event
        await self.emit_event(
            shared,
            ToolStartEvent.create(
                tool=operator_name,
                message=f"Running {operator.description}",
                request_id=shared.get("request_id"),
            ),
        )

        return {
            "operator": operator,
            "operator_name": operator_name,
            "context": context,
            "shared": shared,
        }

    async def exec_async(self, prep_res: dict[str, Any]) -> OperatorResult:
        """Execute the operator."""
        operator = prep_res["operator"]
        context = prep_res["context"]
        shared = prep_res["shared"]

        # Get MCP session if needed
        mcp_session = None
        if operator.requires_mcp:
            session_manager = shared.get("mcp", {}).get("session_manager")
            if session_manager:
                # Create callbacks for event streaming
                callbacks = self._create_callbacks(shared)

                # Get session for operator's tools
                if operator.mcp_tools:
                    tool_name = operator.mcp_tools[0]
                    async with session_manager.session_for_tool(
                        tool_name, callbacks
                    ) as session:
                        return await operator.execute(context, session)
                else:
                    logger.warning("Operator requires MCP but no tools specified")

        # Execute operator (may be pure LLM)
        return await operator.execute(context, mcp_session)

    async def post_async(
        self,
        shared: dict[str, Any],
        prep_res: dict[str, Any],
        exec_res: OperatorResult,
    ) -> str | None:
        """Store result and emit completion event."""
        operator_name = prep_res["operator_name"]

        # Store result
        shared.setdefault("results", {})
        shared["results"].setdefault("tool_outputs", []).append(exec_res)

        # Emit completion or error event
        if exec_res.success:
            await self.emit_event(
                shared,
                ToolCompleteEvent.create(
                    tool=operator_name,
                    content_count=len(exec_res.contents) + (1 if exec_res.output else 0),
                    request_id=shared.get("request_id"),
                ),
            )
        else:
            await self.emit_event(
                shared,
                ToolErrorEvent.create(
                    tool=operator_name,
                    error=exec_res.error or "Unknown error",
                    request_id=shared.get("request_id"),
                ),
            )

        logger.info(
            "Tool execution complete",
            operator=operator_name,
            success=exec_res.success,
        )

        return "default"

    async def exec_fallback_async(self, prep_res: Any, exc: Exception) -> OperatorResult:
        """Handle execution failure."""
        operator_name = prep_res.get("operator_name", "unknown")
        logger.error(f"Tool execution failed: {operator_name}", error=str(exc))

        return OperatorResult.error_result(
            error=f"Operator {operator_name} failed: {str(exc)}"
        )

    def _create_callbacks(self, shared: dict[str, Any]) -> MCPCallbacks:
        """Create MCP callbacks for event streaming."""
        from agentic_chatbot.events.models import ToolProgressEvent, ToolContentEvent

        async def on_progress(
            server_id: str, tool_name: str, progress: float, message: str
        ) -> None:
            await self.emit_event(
                shared,
                ToolProgressEvent.create(
                    tool=tool_name,
                    progress=progress,
                    message=message,
                    request_id=shared.get("request_id"),
                ),
            )

        async def on_content(
            server_id: str, tool_name: str, content: Any, content_type: str
        ) -> None:
            await self.emit_event(
                shared,
                ToolContentEvent.create(
                    tool=tool_name,
                    content_type=content_type,
                    data=content,
                    request_id=shared.get("request_id"),
                ),
            )

        async def on_error(
            server_id: str, tool_name: str, error: str, error_type: str
        ) -> None:
            await self.emit_event(
                shared,
                ToolErrorEvent.create(
                    tool=tool_name,
                    error=error,
                    error_type=error_type,
                    request_id=shared.get("request_id"),
                ),
            )

        return MCPCallbacks(
            on_progress=on_progress,
            on_content=on_content,
            on_error=on_error,
        )
