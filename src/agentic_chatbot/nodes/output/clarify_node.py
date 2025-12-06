"""Clarify node for requesting user input."""

from typing import Any

from agentic_chatbot.config.prompts import CLARIFY_SYSTEM_PROMPT, CLARIFY_PROMPT
from agentic_chatbot.events.models import ClarifyRequestEvent
from agentic_chatbot.nodes.base import AsyncBaseNode
from agentic_chatbot.utils.llm import LLMClient
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class ClarifyNode(AsyncBaseNode):
    """
    Generate clarification question for user.

    Type: Output Node

    When the request is ambiguous, this node generates
    a helpful clarifying question.
    """

    node_name = "clarify"
    description = "Generate clarification question"

    async def prep_async(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Get decision context."""
        decision = shared.get("supervisor", {}).get("current_decision")

        if decision and decision.question:
            # Supervisor already provided question
            return {
                "question": decision.question,
                "from_decision": True,
            }

        # Need to generate question
        return {
            "query": shared.get("user_query", ""),
            "reason": decision.reasoning if decision else "Request is ambiguous",
            "from_decision": False,
        }

    async def exec_async(self, prep_res: dict[str, Any]) -> str:
        """Generate or use clarification question."""
        if prep_res.get("from_decision"):
            return prep_res["question"]

        # Generate question using LLM
        client = LLMClient()

        prompt = CLARIFY_PROMPT.format(
            query=prep_res["query"],
            reason=prep_res["reason"],
        )

        try:
            response = await client.complete(
                prompt=prompt,
                system=CLARIFY_SYSTEM_PROMPT,
                model="haiku",  # Fast for simple generation
            )
            return response.content

        except Exception as e:
            logger.warning(f"Clarification generation failed: {e}")
            return "Could you please provide more details about what you're looking for?"

    async def post_async(
        self,
        shared: dict[str, Any],
        prep_res: dict[str, Any],
        exec_res: str,
    ) -> str | None:
        """Store and emit clarification."""
        # Store as final response
        shared.setdefault("results", {})
        shared["results"]["final_response"] = exec_res
        shared["results"]["is_clarification"] = True

        # Emit clarify event
        await self.emit_event(
            shared,
            ClarifyRequestEvent.create(
                question=exec_res,
                request_id=shared.get("request_id"),
            ),
        )

        logger.info("Clarification requested")

        return "default"
