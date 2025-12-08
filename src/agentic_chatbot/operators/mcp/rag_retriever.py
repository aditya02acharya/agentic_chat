"""RAG retriever operator."""

import time
from typing import TYPE_CHECKING

from ..base import BaseOperator, OperatorType
from ..context import OperatorContext, OperatorResult
from ..registry import OperatorRegistry
from ...core.exceptions import OperatorError

if TYPE_CHECKING:
    from ...mcp.session import MCPSession


@OperatorRegistry.register("rag_retriever")
class RAGRetrieverOperator(BaseOperator):
    """Searches internal documents via RAG."""

    name = "rag_retriever"
    description = "Search internal knowledge base using RAG"
    operator_type = OperatorType.MCP_BACKED
    mcp_tools = ["rag_search", "rag_get_document"]
    context_requirements = ["query"]

    async def execute(
        self,
        context: OperatorContext,
        mcp_session: "MCPSession | None" = None,
    ) -> OperatorResult:
        start_time = time.time()

        if not mcp_session:
            return OperatorResult(
                success=False,
                error="MCP session required for RAG retrieval",
                duration_ms=(time.time() - start_time) * 1000,
            )

        try:
            top_k = context.get_param("top_k", 5)
            result = await mcp_session.call_tool(
                "rag_search",
                {"query": context.query, "top_k": top_k},
            )

            if not result.success:
                return OperatorResult(
                    success=False,
                    error=result.error or "RAG search failed",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            return OperatorResult(
                success=True,
                output=result.text,
                metadata={
                    "tool": "rag_search",
                    "top_k": top_k,
                },
                duration_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return OperatorResult(
                success=False,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )
