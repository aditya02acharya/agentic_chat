"""Query rewriter operator."""

import time
from typing import TYPE_CHECKING

from ..base import BaseOperator, OperatorType
from ..context import OperatorContext, OperatorResult
from ..registry import OperatorRegistry
from ...config.prompts import QUERY_REWRITER_SYSTEM_PROMPT
from ...utils.llm import LLMClient

if TYPE_CHECKING:
    from ...mcp.session import MCPSession


@OperatorRegistry.register("query_rewriter")
class QueryRewriterOperator(BaseOperator):
    """Rewrites queries for better search results."""

    name = "query_rewriter"
    description = "Rewrites and expands queries for better search"
    operator_type = OperatorType.PURE_LLM
    model = "haiku"
    context_requirements = ["query"]

    def __init__(self):
        self._llm = LLMClient()

    async def execute(
        self,
        context: OperatorContext,
        mcp_session: "MCPSession | None" = None,
    ) -> OperatorResult:
        start_time = time.time()

        try:
            response = await self._llm.complete(
                prompt=f"Rewrite this query for search optimization: {context.query}",
                system=QUERY_REWRITER_SYSTEM_PROMPT,
                model=self.model or "haiku",
            )

            return OperatorResult(
                success=True,
                output=response.content,
                metadata={
                    "original_query": context.query,
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
