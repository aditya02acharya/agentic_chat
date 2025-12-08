"""Web searcher operator."""

import time
from typing import TYPE_CHECKING

from ..base import BaseOperator, OperatorType
from ..context import OperatorContext, OperatorResult
from ..registry import OperatorRegistry

if TYPE_CHECKING:
    from ...mcp.session import MCPSession


@OperatorRegistry.register("web_searcher")
class WebSearcherOperator(BaseOperator):
    """Searches the web for information."""

    name = "web_searcher"
    description = "Search the web for current information"
    operator_type = OperatorType.MCP_BACKED
    mcp_tools = ["web_search", "news_search"]
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
                error="MCP session required for web search",
                duration_ms=(time.time() - start_time) * 1000,
            )

        try:
            max_results = context.get_param("max_results", 5)
            result = await mcp_session.call_tool(
                "web_search",
                {"query": context.query, "max_results": max_results},
            )

            if not result.success:
                return OperatorResult(
                    success=False,
                    error=result.error or "Web search failed",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            return OperatorResult(
                success=True,
                output=result.text,
                metadata={
                    "tool": "web_search",
                    "max_results": max_results,
                },
                duration_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return OperatorResult(
                success=False,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )
