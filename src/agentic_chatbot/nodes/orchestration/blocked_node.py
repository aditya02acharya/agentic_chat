"""Handle blocked node for graceful degradation."""

from typing import Any

from agentic_chatbot.config.prompts import BLOCKED_HANDLER_PROMPT
from agentic_chatbot.nodes.base import AsyncBaseNode
from agentic_chatbot.utils.llm import LLMClient
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class HandleBlockedNode(AsyncBaseNode):
    """
    Handle blocked state with graceful degradation.

    Type: Orchestration Node

    When the system cannot proceed (blocked state),
    this node generates a helpful response explaining
    the limitation and suggesting alternatives.
    """

    node_name = "handle_blocked"
    description = "Handle blocked state gracefully"

    async def prep_async(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Prepare context for blocked handling."""
        # Get action history
        action_history = shared.get("supervisor", {}).get("state")
        actions_attempted = []
        if action_history and hasattr(action_history, "action_history"):
            actions_attempted = [
                f"{a.action.value}: {a.decision.reasoning[:50]}..."
                for a in action_history.action_history
            ]

        # Get reflection issues
        reflection = shared.get("reflection", {})
        issues = reflection.get("issues", ["Unknown issue"])

        return {
            "query": shared.get("user_query", ""),
            "actions_attempted": actions_attempted,
            "issues": issues,
        }

    async def exec_async(self, prep_res: dict[str, Any]) -> str:
        """Generate helpful blocked response."""
        client = LLMClient()

        prompt = BLOCKED_HANDLER_PROMPT.format(
            query=prep_res["query"],
            actions_attempted="\n".join(prep_res["actions_attempted"]) or "No actions taken",
            issue="; ".join(prep_res["issues"]),
        )

        try:
            response = await client.complete(
                prompt=prompt,
                model="haiku",  # Fast for error handling
            )
            return response.content

        except Exception as e:
            logger.error(f"Blocked handler failed: {e}")
            # Fallback message
            return (
                "I apologize, but I encountered an issue while processing your request. "
                f"The problem was: {'; '.join(prep_res['issues'])}. "
                "Could you try rephrasing your question or providing more context?"
            )

    async def post_async(
        self,
        shared: dict[str, Any],
        prep_res: dict[str, Any],
        exec_res: str,
    ) -> str | None:
        """Store blocked response."""
        shared.setdefault("results", {})
        shared["results"]["final_response"] = exec_res
        shared["results"]["blocked"] = True

        logger.info("Handled blocked state", issues=prep_res["issues"])

        return "default"
