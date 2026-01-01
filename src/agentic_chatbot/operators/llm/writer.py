"""Writer operator for formatting final responses."""

from typing import TYPE_CHECKING

from agentic_chatbot.config.prompts import WRITER_SYSTEM_PROMPT, WRITER_PROMPT
from agentic_chatbot.operators.base import BaseOperator, OperatorType
from agentic_chatbot.operators.context import OperatorContext, OperatorResult
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.utils.llm import LLMClient

# New unified data model
from agentic_chatbot.data.execution import ExecutionInput, ExecutionOutput
from agentic_chatbot.data.content import TextContent

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

    async def run(
        self,
        input: ExecutionInput,
        mcp_session: "MCPSession | None" = None,
    ) -> ExecutionOutput:
        """
        Execute response formatting using unified data model.

        Args:
            input: ExecutionInput with query and content to format
            mcp_session: Not used (pure LLM operator)

        Returns:
            ExecutionOutput with formatted response as ContentBlock
        """
        client = LLMClient()

        # Get content to format
        content = input.get("content", "")
        if not content and input.step_results:
            # Use step results if content not explicitly provided
            content = "\n\n".join(v.text for v in input.step_results.values())

        prompt = WRITER_PROMPT.format(
            query=input.query,
            content=content,
        )

        try:
            response = await client.complete(
                prompt=prompt,
                system=WRITER_SYSTEM_PROMPT,
                model=self.model or "sonnet",
            )

            return ExecutionOutput.success(
                TextContent.markdown(response.content),
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                metadata={"model": response.model},
            )

        except Exception as e:
            return ExecutionOutput.error(f"Writing failed: {str(e)}")

    async def execute(
        self,
        context: OperatorContext,
        mcp_session: "MCPSession | None" = None,
    ) -> OperatorResult:
        """
        Legacy interface - converts to new run() interface.

        DEPRECATED: Use run() with ExecutionInput instead.
        """
        exec_input = ExecutionInput(
            query=context.query,
            extra=context.extra,
            step_results={},
        )

        output = await self.run(exec_input, mcp_session)

        if output.success:
            return OperatorResult.success_result(
                output=output.text,
                input_tokens=output.input_tokens,
                output_tokens=output.output_tokens,
                metadata=output.metadata,
            )
        else:
            return OperatorResult.error_result(error=output.error)
