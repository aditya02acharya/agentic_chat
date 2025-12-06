"""Dynamic context assembler using Builder pattern."""

from typing import Any

from agentic_chatbot.context.memory import ConversationMemory
from agentic_chatbot.context.results import ResultStore
from agentic_chatbot.context.actions import ActionHistory
from agentic_chatbot.mcp.registry import MCPServerRegistry
from agentic_chatbot.operators.context import OperatorContext
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class ContextAssembler:
    """
    Dynamically assembles context based on component requirements.

    Design Pattern: Builder Pattern

    Components declare what context they need via a requirements DSL,
    and this assembler pulls the relevant data from various sources.

    Requirements DSL:
    - "conversation.recent(5)"     → Last 5 messages
    - "conversation.summary"       → Summary of older messages
    - "tools.summaries"            → All tool short descriptions
    - "tools.schema(web_search)"   → Full schema (lazy loaded)
    - "results.step(2)"            → Output from workflow step 2
    - "results.all"                → All results this turn
    - "actions.this_turn"          → Actions taken this turn
    """

    def __init__(
        self,
        memory: ConversationMemory,
        mcp_registry: MCPServerRegistry,
        result_store: ResultStore,
        action_history: ActionHistory,
    ):
        """
        Initialize context assembler.

        Args:
            memory: Conversation memory
            mcp_registry: MCP server/tool registry
            result_store: Result store for this turn
            action_history: Action history
        """
        self._sources = {
            "conversation": memory,
            "tools": mcp_registry,
            "results": result_store,
            "actions": action_history,
        }

    async def assemble(
        self,
        requirements: list[str],
        query: str,
        shared_store: dict[str, Any] | None = None,
    ) -> OperatorContext:
        """
        Assemble context based on requirements.

        Args:
            requirements: List of requirement strings
            query: The user's query
            shared_store: Optional shared store reference

        Returns:
            Assembled OperatorContext
        """
        context = OperatorContext(
            query=query,
            shared_store=shared_store or {},
        )

        for req in requirements:
            try:
                data = await self._resolve_requirement(req)
                self._apply_to_context(context, req, data)
            except Exception as e:
                logger.warning(f"Failed to resolve requirement '{req}': {e}")

        return context

    async def _resolve_requirement(self, req: str) -> Any:
        """
        Resolve a single requirement.

        Args:
            req: Requirement string (e.g., "conversation.recent(5)")

        Returns:
            Resolved data
        """
        # Parse requirement
        if "." not in req:
            return None

        parts = req.split(".", 1)
        source_name = parts[0]
        method = parts[1] if len(parts) > 1 else "all"

        source = self._sources.get(source_name)
        if source is None:
            logger.warning(f"Unknown source: {source_name}")
            return None

        # Special handling for tools.schema
        if source_name == "tools" and method.startswith("schema("):
            tool_name = method[7:-1]  # Extract from "schema(tool_name)"
            return await self._sources["tools"].get_tool_schema(tool_name)

        # Special handling for tools.summaries
        if source_name == "tools" and method == "summaries":
            return await self._sources["tools"].get_all_tool_summaries()

        # Generic resolution via source's get method
        if hasattr(source, "get"):
            return await source.get(method)

        return None

    def _apply_to_context(
        self,
        context: OperatorContext,
        req: str,
        data: Any,
    ) -> None:
        """
        Apply resolved data to context.

        Args:
            context: Context to update
            req: Original requirement string
            data: Resolved data
        """
        if data is None:
            return

        # Map requirements to context fields
        if req.startswith("conversation.recent"):
            if hasattr(data, "__iter__") and not isinstance(data, (str, dict)):
                context.recent_messages = [
                    {"role": m.role, "content": m.content}
                    if hasattr(m, "role") else m
                    for m in data
                ]
        elif req == "conversation.summary":
            context.conversation_summary = str(data)
        elif req.startswith("tools.schema"):
            tool_name = req.split("(")[1].rstrip(")")
            context.tool_schemas[tool_name] = (
                data.model_dump() if hasattr(data, "model_dump") else data
            )
        elif req.startswith("results.step"):
            step_id = req.split("(")[1].rstrip(")")
            if data:
                context.step_results[step_id] = (
                    data.output if hasattr(data, "output") else data
                )
        elif req == "results.all":
            if isinstance(data, dict):
                context.step_results.update(data)
        else:
            # Store in extra for custom requirements
            context.extra[req] = data

    async def assemble_supervisor_context(
        self,
        query: str,
        shared_store: dict[str, Any] | None = None,
    ) -> OperatorContext:
        """
        Assemble context specifically for supervisor decisions.

        Args:
            query: User's query
            shared_store: Optional shared store

        Returns:
            Supervisor-optimized context
        """
        requirements = [
            "conversation.recent(5)",
            "conversation.summary",
            "tools.summaries",
            "actions.this_turn",
        ]
        return await self.assemble(requirements, query, shared_store)

    async def assemble_operator_context(
        self,
        operator_name: str,
        operator_requirements: list[str],
        query: str,
        shared_store: dict[str, Any] | None = None,
    ) -> OperatorContext:
        """
        Assemble context for a specific operator.

        Args:
            operator_name: Name of the operator
            operator_requirements: Operator's declared requirements
            query: User's query
            shared_store: Optional shared store

        Returns:
            Operator-specific context
        """
        # Always include query
        requirements = list(operator_requirements)
        if "query" not in requirements:
            requirements.insert(0, "query")

        return await self.assemble(requirements, query, shared_store)
