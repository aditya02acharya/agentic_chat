"""Synthesizer operator for combining multiple sources."""

from typing import TYPE_CHECKING, Any

from agentic_chatbot.config.prompts import SYNTHESIZER_SYSTEM_PROMPT, SYNTHESIZER_PROMPT
from agentic_chatbot.operators.base import BaseOperator, OperatorType
from agentic_chatbot.operators.context import OperatorContext, OperatorResult
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.utils.llm import LLMClient

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

    async def execute(
        self,
        context: OperatorContext,
        mcp_session: "MCPSession | None" = None,
    ) -> OperatorResult:
        """
        Execute synthesis of multiple sources.

        Args:
            context: Operator context with query and source results
            mcp_session: Not used (pure LLM operator)

        Returns:
            OperatorResult with synthesized content
        """
        client = LLMClient()

        # Format sources for the prompt
        sources = context.step_results or context.extra.get("sources", {})
        sources_text = self._format_sources(sources)

        prompt = SYNTHESIZER_PROMPT.format(
            query=context.query,
            sources=sources_text,
        )

        try:
            response = await client.complete(
                prompt=prompt,
                system=SYNTHESIZER_SYSTEM_PROMPT,
                model=self.model or "sonnet",
            )

            return OperatorResult.success_result(
                output=response.content,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                metadata={
                    "source_count": len(sources),
                    "model": response.model,
                },
            )

        except Exception as e:
            return OperatorResult.error_result(
                error=f"Synthesis failed: {str(e)}",
            )

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
