"""Writer operator for formatting final responses."""

import time
from typing import TYPE_CHECKING

from ..base import BaseOperator, OperatorType
from ..context import OperatorContext, OperatorResult
from ..registry import OperatorRegistry
from ...config.prompts import WRITER_SYSTEM_PROMPT
from ...utils.llm import LLMClient

if TYPE_CHECKING:
    from ...mcp.session import MCPSession


@OperatorRegistry.register("writer")
class WriterOperator(BaseOperator):
    """Formats final responses for users."""

    name = "writer"
    description = "Formats content into user-friendly responses"
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
            content = context.get_param("content", "")
            if not content and context.previous_results:
                content = "\n".join(str(r) for r in context.previous_results)

            prompt = f"""User Question: {context.query}

Content to format:
{content}

Please write a clear, helpful response that directly addresses the user's question."""

            response = await self._llm.complete(
                prompt=prompt,
                system=WRITER_SYSTEM_PROMPT,
                model=self.model or "sonnet",
            )

            return OperatorResult(
                success=True,
                output=response.content,
                metadata={
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
