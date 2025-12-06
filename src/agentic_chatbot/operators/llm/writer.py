"""Writer operator for formatting final responses."""

from typing import TYPE_CHECKING

from agentic_chatbot.config.prompts import WRITER_SYSTEM_PROMPT, WRITER_PROMPT
from agentic_chatbot.operators.base import BaseOperator, OperatorType
from agentic_chatbot.operators.context import OperatorContext, OperatorResult
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.utils.llm import LLMClient

if TYPE_CHECKING:
    from agentic_chatbot.mcp.session import MCPSession


@OperatorRegistry.register("writer")
class WriterOperator(BaseOperator):
    """
    Formats content as a clear, helpful response for users.

    Type: PURE_LLM
    Model: Sonnet (needs good writing quality)

    Takes raw content and formats it appropriately for the user,
    using markdown and proper structure.
    """

    name = "writer"
    description = "Formats content as a helpful user response"
    operator_type = OperatorType.PURE_LLM
    model = "sonnet"
    context_requirements = ["query", "content"]

    async def execute(
        self,
        context: OperatorContext,
        mcp_session: "MCPSession | None" = None,
    ) -> OperatorResult:
        """
        Execute response formatting.

        Args:
            context: Operator context with query and content to format
            mcp_session: Not used (pure LLM operator)

        Returns:
            OperatorResult with formatted response
        """
        client = LLMClient()

        # Get content to format
        content = context.extra.get("content", "")
        if not content and context.step_results:
            # Use step results if content not explicitly provided
            content = "\n\n".join(str(v) for v in context.step_results.values())

        prompt = WRITER_PROMPT.format(
            query=context.query,
            content=content,
        )

        try:
            response = await client.complete(
                prompt=prompt,
                system=WRITER_SYSTEM_PROMPT,
                model=self.model or "sonnet",
            )

            return OperatorResult.success_result(
                output=response.content,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                metadata={"model": response.model},
            )

        except Exception as e:
            return OperatorResult.error_result(
                error=f"Writing failed: {str(e)}",
            )
