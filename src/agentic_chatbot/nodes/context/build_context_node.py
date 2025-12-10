"""Build context node for assembling operator context."""

from typing import Any

from agentic_chatbot.context.assembler import ContextAssembler
from agentic_chatbot.nodes.base import AsyncBaseNode
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class BuildContextNode(AsyncBaseNode):
    """
    Build focused context for operator execution.

    Type: Context Node

    Uses ContextAssembler to create operator-specific
    context based on the operator's declared requirements.
    """

    node_name = "build_context"
    description = "Build focused context for operator"

    async def prep_async(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Get operator and assembler."""
        decision = shared.get("supervisor", {}).get("current_decision")
        operator_name = decision.operator if decision else None

        # Get context assembler components
        memory = shared.get("memory")
        result_store = shared.get("result_store")
        action_history = shared.get("action_history")
        mcp_registry = shared.get("mcp", {}).get("server_registry")

        return {
            "operator_name": operator_name,
            "query": shared.get("user_query", ""),
            "memory": memory,
            "result_store": result_store,
            "action_history": action_history,
            "mcp_registry": mcp_registry,
        }

    async def exec_async(self, prep_res: dict[str, Any]) -> dict[str, Any]:
        """Build context using assembler."""
        operator_name = prep_res.get("operator_name")

        if not operator_name:
            return {"context": None, "error": "No operator specified"}

        # Get operator requirements
        try:
            operator_class = OperatorRegistry.get(operator_name)
            requirements = getattr(operator_class, "context_requirements", [])
        except KeyError:
            return {"context": None, "error": f"Unknown operator: {operator_name}"}

        # Check if we have all components for assembler
        memory = prep_res.get("memory")
        result_store = prep_res.get("result_store")
        action_history = prep_res.get("action_history")
        mcp_registry = prep_res.get("mcp_registry")

        if not all([memory, result_store, action_history]):
            # Minimal context without full assembler
            from agentic_chatbot.operators.context import OperatorContext

            return {
                "context": OperatorContext(query=prep_res["query"]),
                "minimal": True,
            }

        # Use full assembler
        assembler = ContextAssembler(
            memory=memory,
            mcp_registry=mcp_registry,
            result_store=result_store,
            action_history=action_history,
        )

        context = await assembler.assemble_operator_context(
            operator_name=operator_name,
            operator_requirements=requirements,
            query=prep_res["query"],
        )

        return {"context": context, "requirements": requirements}

    async def post_async(
        self,
        shared: dict[str, Any],
        prep_res: dict[str, Any],
        exec_res: dict[str, Any],
    ) -> str | None:
        """Store built context."""
        shared["operator_context"] = exec_res.get("context")

        if exec_res.get("error"):
            logger.warning("Context build error", error=exec_res["error"])
        else:
            logger.debug(
                "Context built",
                operator=prep_res.get("operator_name"),
                requirements=exec_res.get("requirements", []),
            )

        return "default"
