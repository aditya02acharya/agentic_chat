"""Initialize node for setting up request context."""

from typing import Any

from agentic_chatbot.context.memory import ConversationMemory
from agentic_chatbot.context.results import ResultStore
from agentic_chatbot.context.actions import ActionHistory
from agentic_chatbot.core.supervisor import SupervisorState
from agentic_chatbot.nodes.base import AsyncBaseNode
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class InitializeNode(AsyncBaseNode):
    """
    Initialize request context and shared store.

    Type: Context Node

    Sets up all required components in the shared store
    for the chat flow to operate.
    """

    node_name = "initialize"
    description = "Initialize request context"

    async def prep_async(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Get initialization parameters."""
        return {
            "conversation_id": shared.get("conversation_id", ""),
            "request_id": shared.get("request_id", ""),
            "user_query": shared.get("user_query", ""),
        }

    async def exec_async(self, prep_res: dict[str, Any]) -> dict[str, Any]:
        """Create context components."""
        # Create fresh components for this request
        return {
            "memory": ConversationMemory(window_size=5),
            "result_store": ResultStore(),
            "action_history": ActionHistory(),
            "supervisor_state": SupervisorState(max_iterations=5),
        }

    async def post_async(
        self,
        shared: dict[str, Any],
        prep_res: dict[str, Any],
        exec_res: dict[str, Any],
    ) -> str | None:
        """Store components in shared."""
        # Initialize memory (may already exist from previous turns)
        if "memory" not in shared:
            shared["memory"] = exec_res["memory"]

        # Always create fresh for new request
        shared["result_store"] = exec_res["result_store"]
        shared["action_history"] = exec_res["action_history"]

        # Supervisor state
        shared.setdefault("supervisor", {})
        shared["supervisor"]["state"] = exec_res["supervisor_state"]
        shared["supervisor"]["current_decision"] = None

        # Results container
        shared["results"] = {
            "tool_outputs": [],
            "workflow_output": None,
            "synthesis": None,
            "final_response": None,
        }

        # Reflection container
        shared["reflection"] = {}

        # Add user message to memory
        memory = shared["memory"]
        if hasattr(memory, "add_message"):
            memory.add_message("user", prep_res["user_query"])

        logger.info(
            "Request initialized",
            conversation_id=prep_res["conversation_id"],
            request_id=prep_res["request_id"],
        )

        return "default"
