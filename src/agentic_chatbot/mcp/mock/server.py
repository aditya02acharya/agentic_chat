"""Mock MCP server for local testing."""

import asyncio
from typing import Any

from agentic_chatbot.mcp.mock.tools import MockTool, create_default_mock_tools
from agentic_chatbot.mcp.models import ToolResult, ToolResultStatus
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class MockMCPServer:
    """
    Mock MCP server for testing without real MCP servers.

    This can be used directly or via a FastAPI app for HTTP testing.
    """

    def __init__(
        self,
        server_id: str = "mock_server",
        tools: list[MockTool] | None = None,
    ):
        """
        Initialize mock server.

        Args:
            server_id: Server identifier
            tools: List of mock tools (uses defaults if not provided)
        """
        self.server_id = server_id
        self._tools: dict[str, MockTool] = {}

        for tool in tools or create_default_mock_tools():
            self._tools[tool.name] = tool

    @property
    def tool_names(self) -> list[str]:
        """Get list of tool names."""
        return list(self._tools.keys())

    def get_tool_list(self) -> dict[str, Any]:
        """Get tool list response."""
        return {
            "server_id": self.server_id,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                }
                for tool in self._tools.values()
            ],
        }

    def get_tool_schema(self, tool_name: str) -> dict[str, Any] | None:
        """Get tool schema."""
        tool = self._tools.get(tool_name)
        if not tool:
            return None

        return {
            "name": tool.name,
            "description": tool.description,
            "server_id": self.server_id,
            "input_schema": tool.input_schema,
        }

    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """
        Call a tool.

        Args:
            tool_name: Name of the tool
            params: Tool parameters

        Returns:
            Tool result
        """
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                status=ToolResultStatus.ERROR,
                error=f"Tool '{tool_name}' not found",
            )

        try:
            logger.debug("Mock executing tool", tool_name=tool_name, params=params)
            result = await tool.execute(params)
            return result
        except Exception as e:
            logger.error("Mock tool execution failed", tool_name=tool_name, error=str(e))
            return ToolResult(
                tool_name=tool_name,
                status=ToolResultStatus.ERROR,
                error=str(e),
            )

    def health_check(self) -> dict[str, Any]:
        """Health check response."""
        return {
            "status": "ok",
            "server_id": self.server_id,
            "tools": len(self._tools),
        }


def create_mock_discovery_response(servers: list[MockMCPServer]) -> dict[str, Any]:
    """
    Create mock discovery service response.

    Args:
        servers: List of mock servers

    Returns:
        Discovery response
    """
    return {
        "servers": [
            {
                "id": server.server_id,
                "name": f"Mock {server.server_id}",
                "url": f"http://localhost:8080/{server.server_id}",
                "description": f"Mock MCP server: {server.server_id}",
                "version": "1.0.0",
                "tools": server.tool_names,
                "healthy": True,
            }
            for server in servers
        ]
    }
