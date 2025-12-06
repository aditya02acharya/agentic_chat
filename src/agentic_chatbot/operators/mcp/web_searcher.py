"""Web searcher operator for external search."""

from typing import TYPE_CHECKING

from agentic_chatbot.core.exceptions import OperatorError
from agentic_chatbot.operators.base import BaseOperator, OperatorType
from agentic_chatbot.operators.context import OperatorContext, OperatorResult
from agentic_chatbot.operators.registry import OperatorRegistry

if TYPE_CHECKING:
    from agentic_chatbot.mcp.session import MCPSession


@OperatorRegistry.register("web_searcher")
class WebSearcherOperator(BaseOperator):
    """
    Searches the web for information.

    Type: MCP_BACKED (no LLM, just MCP tool calls)
    MCP Tools: web_search, news_search

    Retrieves information from the web based on the search query.
    """

    name = "web_searcher"
    description = "Search the web for information"
    operator_type = OperatorType.MCP_BACKED
    mcp_tools = ["web_search", "news_search"]
    context_requirements = ["query", "tools.schema(web_search)"]

    async def execute(
        self,
        context: OperatorContext,
        mcp_session: "MCPSession | None" = None,
    ) -> OperatorResult:
        """
        Execute web search.

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
            # Determine which search tool to use
            search_type = context.extra.get("search_type", "web")
            tool_name = "news_search" if search_type == "news" else "web_search"

            # Call the web search tool
            result = await mcp_session.call_tool(
                tool_name,
                {
                    "query": context.query,
                    "num_results": context.extra.get("num_results", 10),
                },
            )

            if not result.status.value == "success":
                return OperatorResult.error_result(
                    error=result.error or "Web search failed",
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
                    "tool": tool_name,
                    "duration_ms": result.duration_ms,
                },
            )

        except Exception as e:
            return OperatorResult.error_result(
                error=f"Web search failed: {str(e)}",
            )
