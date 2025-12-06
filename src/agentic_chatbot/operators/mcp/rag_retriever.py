"""RAG retriever operator for searching internal documents."""

from typing import TYPE_CHECKING

from agentic_chatbot.core.exceptions import OperatorError
from agentic_chatbot.operators.base import BaseOperator, OperatorType
from agentic_chatbot.operators.context import OperatorContext, OperatorResult
from agentic_chatbot.operators.registry import OperatorRegistry

if TYPE_CHECKING:
    from agentic_chatbot.mcp.session import MCPSession


@OperatorRegistry.register("rag_retriever")
class RAGRetrieverOperator(BaseOperator):
    """
    Searches internal documents via RAG.

    Type: MCP_BACKED (no LLM, just MCP tool calls)
    MCP Tools: rag_search, rag_get_document

    Retrieves relevant documents from the internal knowledge base
    based on the search query.
    """

    name = "rag_retriever"
    description = "Search internal knowledge base"
    operator_type = OperatorType.MCP_BACKED
    mcp_tools = ["rag_search", "rag_get_document"]
    context_requirements = ["query", "tools.schema(rag_search)"]

    # Fallback to web search if RAG fails
    fallback_operator = "web_searcher"

    async def execute(
        self,
        context: OperatorContext,
        mcp_session: "MCPSession | None" = None,
    ) -> OperatorResult:
        """
        Execute RAG search.

        Args:
            context: Operator context with query
            mcp_session: Required MCP session

        Returns:
            OperatorResult with search results

        Raises:
            OperatorError: If MCP session not provided
        """
        if not mcp_session:
            raise OperatorError("MCP session required", operator_name=self.name)

        try:
            # Call the RAG search tool
            result = await mcp_session.call_tool(
                "rag_search",
                {
                    "query": context.query,
                    "top_k": context.extra.get("top_k", 5),
                },
            )

            if not result.status.value == "success":
                return OperatorResult.error_result(
                    error=result.error or "RAG search failed",
                )

            # Extract text content
            output = result.combined_text
            if not output and result.contents:
                # Handle JSON response
                for content in result.contents:
                    if content.content_type == "application/json":
                        output = content.data
                        break

            return OperatorResult.success_result(
                output=output,
                contents=result.contents,
                metadata={
                    "tool": "rag_search",
                    "duration_ms": result.duration_ms,
                },
            )

        except Exception as e:
            return OperatorResult.error_result(
                error=f"RAG retrieval failed: {str(e)}",
            )
