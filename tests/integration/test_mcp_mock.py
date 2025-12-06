"""Integration tests for MCP mock server."""

import pytest

from agentic_chatbot.mcp.mock.server import MockMCPServer, create_mock_discovery_response
from agentic_chatbot.mcp.mock.tools import create_default_mock_tools
from agentic_chatbot.mcp.models import ToolResultStatus


class TestMockMCPServer:
    """Tests for mock MCP server."""

    @pytest.fixture
    def server(self):
        """Create mock server."""
        return MockMCPServer(server_id="test_server")

    def test_tool_list(self, server):
        """Test getting tool list."""
        tool_list = server.get_tool_list()

        assert tool_list["server_id"] == "test_server"
        assert len(tool_list["tools"]) > 0
        assert any(t["name"] == "rag_search" for t in tool_list["tools"])

    def test_get_tool_schema(self, server):
        """Test getting tool schema."""
        schema = server.get_tool_schema("rag_search")

        assert schema is not None
        assert schema["name"] == "rag_search"
        assert "input_schema" in schema

    def test_get_unknown_tool_schema(self, server):
        """Test getting unknown tool schema."""
        schema = server.get_tool_schema("nonexistent")
        assert schema is None

    @pytest.mark.asyncio
    async def test_call_rag_search(self, server):
        """Test calling RAG search tool."""
        result = await server.call_tool("rag_search", {"query": "test query"})

        assert result.status == ToolResultStatus.SUCCESS
        assert len(result.contents) > 0

    @pytest.mark.asyncio
    async def test_call_web_search(self, server):
        """Test calling web search tool."""
        result = await server.call_tool("web_search", {"query": "test"})

        assert result.status == ToolResultStatus.SUCCESS
        assert result.combined_text != ""

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self, server):
        """Test calling unknown tool."""
        result = await server.call_tool("nonexistent", {})

        assert result.status == ToolResultStatus.ERROR
        assert "not found" in result.error.lower()

    def test_health_check(self, server):
        """Test health check."""
        health = server.health_check()

        assert health["status"] == "ok"
        assert health["server_id"] == "test_server"


class TestMockDiscovery:
    """Tests for mock discovery response."""

    def test_create_discovery_response(self):
        """Test creating discovery response."""
        servers = [
            MockMCPServer("server1"),
            MockMCPServer("server2"),
        ]

        response = create_mock_discovery_response(servers)

        assert len(response["servers"]) == 2
        assert response["servers"][0]["id"] == "server1"
        assert response["servers"][1]["id"] == "server2"
