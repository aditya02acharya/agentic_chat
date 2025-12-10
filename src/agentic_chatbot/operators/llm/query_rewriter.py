"""Query rewriter operator for optimizing search queries."""

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from agentic_chatbot.config.prompts import QUERY_REWRITER_SYSTEM_PROMPT, QUERY_REWRITER_PROMPT
from agentic_chatbot.operators.base import BaseOperator, OperatorType
from agentic_chatbot.operators.context import OperatorContext, OperatorResult
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.utils.structured_llm import StructuredLLMCaller

if TYPE_CHECKING:
    from agentic_chatbot.mcp.session import MCPSession


class RewrittenQuery(BaseModel):
    """Schema for rewritten query output."""

    internal_query: str = Field(..., description="Query optimized for internal RAG search")
    external_query: str = Field(..., description="Query optimized for web search")
    keywords: list[str] = Field(default_factory=list, description="Key terms extracted")


@OperatorRegistry.register("query_rewriter")
class QueryRewriterOperator(BaseOperator):
    """
    Rewrites queries for better search results.

    Type: PURE_LLM (Fast Path)
    Model: Haiku (fast, cost-effective)

    Takes a user query and produces optimized versions for:
    - Internal knowledge base search (RAG)
    - External web search
    - Key term extraction
    """

    name = "query_rewriter"
    description = "Rewrites and expands queries for better search results"
    operator_type = OperatorType.PURE_LLM
    model = "haiku"
    context_requirements = ["query"]

    async def execute(
        self,
        context: OperatorContext,
        mcp_session: "MCPSession | None" = None,
    ) -> OperatorResult:
        """
        Execute query rewriting.

        Args:
            context: Operator context with query
            mcp_session: Not used (pure LLM operator)

        Returns:
            OperatorResult with rewritten queries
        """
        caller = StructuredLLMCaller(max_retries=2)

        prompt = QUERY_REWRITER_PROMPT.format(query=context.query)

        try:
            result = await caller.call(
                prompt=prompt,
                response_model=RewrittenQuery,
                system=QUERY_REWRITER_SYSTEM_PROMPT,
                model=self.model or "haiku",
            )

            return OperatorResult.success_result(
                output={
                    "internal_query": result.internal_query,
                    "external_query": result.external_query,
                    "keywords": result.keywords,
                },
                metadata={"original_query": context.query},
            )

        except Exception as e:
            return OperatorResult.error_result(
                error=f"Query rewriting failed: {str(e)}",
            )
