"""Write node for formatting final response."""

from typing import Any

from agentic_chatbot.nodes.base import AsyncBaseNode
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.operators.context import OperatorContext
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class WriteNode(AsyncBaseNode):
    """
    Format final response for user.

    Type: Output Node
    Uses: Writer operator (Sonnet model)

    Takes the synthesized content or direct answer and
    formats it as a polished user response.
    """

    node_name = "write"
    description = "Format response with Writer operator"

    async def prep_async(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Get content to format."""
        results = shared.get("results", {})

        # Check for different content sources
        content = None

        # Priority: final_response > synthesis > direct answer
        if results.get("final_response"):
            content = results["final_response"]
        elif results.get("synthesis"):
            content = results["synthesis"]
        else:
            # Check for direct answer from supervisor
            decision = shared.get("supervisor", {}).get("current_decision")
            if decision and decision.response:
                content = decision.response

        # If still no content, try tool outputs
        if not content and results.get("tool_outputs"):
            outputs = results["tool_outputs"]
            if outputs:
                last_output = outputs[-1]
                content = last_output.text_output if hasattr(last_output, "text_output") else str(last_output)

        return {
            "content": content,
            "query": shared.get("user_query", ""),
            "needs_formatting": content is not None and len(str(content)) > 100,
        }

    async def exec_async(self, prep_res: dict[str, Any]) -> dict[str, Any]:
        """Format content."""
        content = prep_res.get("content")
        query = prep_res["query"]

        if content is None:
            return {
                "response": "I apologize, but I wasn't able to generate a response. Could you please rephrase your question?",
                "formatted": False,
            }

        # Short responses don't need formatting
        if not prep_res["needs_formatting"]:
            return {"response": str(content), "formatted": False}

        # Use writer operator for longer content
        try:
            writer = OperatorRegistry.create("writer")
        except KeyError:
            # No writer available, return as-is
            return {"response": str(content), "formatted": False}

        context = OperatorContext(query=query)
        context.extra["content"] = content

        try:
            result = await writer.execute(context)
            return {
                "response": result.output if result.success else str(content),
                "formatted": result.success,
            }
        except Exception as e:
            logger.warning(f"Writer failed: {e}")
            return {"response": str(content), "formatted": False}

    async def post_async(
        self,
        shared: dict[str, Any],
        prep_res: dict[str, Any],
        exec_res: dict[str, Any],
    ) -> str | None:
        """Store formatted response."""
        shared.setdefault("results", {})
        shared["results"]["final_response"] = exec_res["response"]

        # Also store in memory for conversation history
        memory = shared.get("memory")
        if memory and hasattr(memory, "add_message"):
            memory.add_message("assistant", exec_res["response"])

        logger.debug(
            "Response written",
            formatted=exec_res["formatted"],
            length=len(exec_res["response"]),
        )

        return "default"
