"""Web searcher operator for external search."""

from typing import TYPE_CHECKING

from agentic_chatbot.core.exceptions import OperatorError
from agentic_chatbot.operators.base import BaseOperator, OperatorType
from agentic_chatbot.operators.context import OperatorContext, OperatorResult
from agentic_chatbot.operators.registry import OperatorRegistry

# New unified data model
from agentic_chatbot.data.execution import ExecutionInput, ExecutionOutput
from agentic_chatbot.data.content import TextContent, ContentBlock

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

    async def run(
        self,
        input: ExecutionInput,
        mcp_session: "MCPSession | None" = None,
    ) -> ExecutionOutput:
        """
        Execute web search using unified data model.

        Args:
            input: ExecutionInput with query
            mcp_session: Required MCP session

        Returns:
            ExecutionOutput with search results as ContentBlock

        Raises:
            OperatorError: If MCP session not provided
        """
        if not mcp_session:
            raise OperatorError("MCP session required", operator_name=self.name)

        try:
            # Determine which search tool to use
            search_type = input.get("search_type", "web")
            tool_name = "news_search" if search_type == "news" else "web_search"

            # Call the web search tool
            result = await mcp_session.call_tool(
                tool_name,
                {
                    "query": input.query,
                    "num_results": input.get("num_results", 10),
                },
            )

            if not result.status.value == "success":
                return ExecutionOutput.error(result.error or "Web search failed")

            # Extract text content
            output = result.combined_text
            if not output and result.contents:
                # Handle JSON response
                for content in result.contents:
                    if content.content_type == "application/json":
                        output = str(content.data)
                        break

            # Convert MCP ToolContent to ContentBlock
            contents = [TextContent.markdown(output)] if output else []

            return ExecutionOutput.success(
                contents[0] if contents else TextContent.plain("No results found"),
                metadata={
                    "tool": tool_name,
                    "duration_ms": result.duration_ms,
                },
            )

        except Exception as e:
            return ExecutionOutput.error(f"Web search failed: {str(e)}")

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

        output = await self.run(exec_input, mcp_session)

        if output.success:
            return OperatorResult.success_result(
                output=output.text,
                metadata=output.metadata,
            )
        else:
            return OperatorResult.error_result(error=output.error)
