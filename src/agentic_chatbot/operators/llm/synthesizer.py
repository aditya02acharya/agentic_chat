"""Synthesizer operator for combining information."""

import time
from typing import TYPE_CHECKING

from ..base import BaseOperator, OperatorType
from ..context import OperatorContext, OperatorResult
from ..registry import OperatorRegistry
from ...config.prompts import SYNTHESIZER_SYSTEM_PROMPT
from ...utils.llm import LLMClient

if TYPE_CHECKING:
    from ...mcp.session import MCPSession


@OperatorRegistry.register("synthesizer")
class SynthesizerOperator(BaseOperator):
    """Synthesizes information from multiple sources."""

    name = "synthesizer"
    description = "Combines multiple sources into a coherent response"
    operator_type = OperatorType.PURE_LLM
    model = "sonnet"
    context_requirements = ["query", "previous_results"]

    def __init__(self):
        self._llm = LLMClient()

    async def execute(
        self,
        context: OperatorContext,
        mcp_session: "MCPSession | None" = None,
    ) -> OperatorResult:
        start_time = time.time()

        try:
            sources_text = "\n\n---\n\n".join(
                str(r) for r in context.previous_results
            )

            prompt = f"""User Query: {context.query}

Information from multiple sources:

{sources_text}

Please synthesize this information into a comprehensive, coherent response."""

            response = await self._llm.complete(
                prompt=prompt,
                system=SYNTHESIZER_SYSTEM_PROMPT,
                model=self.model or "sonnet",
            )

            return OperatorResult(
                success=True,
                output=response.content,
                metadata={
                    "sources_count": len(context.previous_results),
                    "tokens_used": response.input_tokens + response.output_tokens,
                },
                duration_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return OperatorResult(
                success=False,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )
