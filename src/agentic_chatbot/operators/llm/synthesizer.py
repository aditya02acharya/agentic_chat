"""Synthesizer operator for combining multiple sources."""

from typing import TYPE_CHECKING, Any

from agentic_chatbot.config.prompts import SYNTHESIZER_SYSTEM_PROMPT, SYNTHESIZER_PROMPT
from agentic_chatbot.operators.base import BaseOperator, OperatorType
from agentic_chatbot.operators.context import OperatorContext, OperatorResult
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.utils.llm import LLMClient

# New unified data model
from agentic_chatbot.data.execution import ExecutionInput, ExecutionOutput
from agentic_chatbot.data.content import TextContent

if TYPE_CHECKING:
    from agentic_chatbot.mcp.session import MCPSession


@OperatorRegistry.register("synthesizer")
class SynthesizerOperator(BaseOperator):
    """
    Combines information from multiple sources into coherent content.

    Type: PURE_LLM
    Model: Sonnet (needs good reasoning for synthesis)

    Takes multiple source results and produces a unified response
    that addresses the user's original query.
    """

    name = "synthesizer"
    description = "Combines information from multiple sources"
    operator_type = OperatorType.PURE_LLM
    model = "sonnet"
    context_requirements = ["query", "results.all"]

    async def run(
        self,
        input: ExecutionInput,
        mcp_session: "MCPSession | None" = None,
    ) -> ExecutionOutput:
        """
        Execute synthesis of multiple sources using unified data model.

        Args:
            input: ExecutionInput with query and source results
            mcp_session: Not used (pure LLM operator)

        Returns:
            ExecutionOutput with synthesized content as ContentBlock
        """
        client = LLMClient()

        # Format sources for the prompt - use step_results (ExecutionOutput)
        sources = {}
        for k, v in input.step_results.items():
            sources[k] = v.text

        # Also check extra for additional sources
        if not sources:
            sources = input.get("sources", {})

        sources_text = self._format_sources(sources)

        prompt = SYNTHESIZER_PROMPT.format(
            query=input.query,
            sources=sources_text,
        )

        try:
            response = await client.complete(
                prompt=prompt,
                system=SYNTHESIZER_SYSTEM_PROMPT,
                model=self.model or "sonnet",
            )

            return ExecutionOutput.success(
                TextContent.markdown(response.content),
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                metadata={
                    "source_count": len(sources),
                    "model": response.model,
                },
            )

        except Exception as e:
            return ExecutionOutput.error(f"Synthesis failed: {str(e)}")

    def _format_sources(self, sources: dict[str, Any]) -> str:
        """Format sources for the prompt."""
        if not sources:
            return "No sources available."

        formatted = []
        for i, (source_id, content) in enumerate(sources.items(), 1):
            if isinstance(content, dict):
                # Handle structured results
                text = content.get("output", content.get("text", str(content)))
            else:
                text = str(content)

            formatted.append(f"### Source {i} ({source_id})\n{text}")

        return "\n\n".join(formatted)

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
        # Add legacy step_results to extra for compatibility
        if context.step_results:
            exec_input.extra["sources"] = context.step_results

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
