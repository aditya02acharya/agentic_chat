"""Mock MCP server for local testing."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any

from .tools import MOCK_TOOLS, execute_mock_tool


class ToolCallRequest(BaseModel):
    arguments: dict[str, Any] = {}


def create_mock_mcp_app() -> FastAPI:
    """Create a FastAPI app that mimics an MCP server."""
    app = FastAPI(title="Mock MCP Server")

    @app.get("/tools")
    async def list_tools():
        """List available tools."""
        return {
            "tools": [
                {"name": t["name"], "description": t["description"]}
                for t in MOCK_TOOLS.values()
            ]
        }

    @app.get("/tools/{tool_name}")
    async def get_tool(tool_name: str):
        """Get tool schema."""
        if tool_name not in MOCK_TOOLS:
            raise HTTPException(status_code=404, detail="Tool not found")
        return MOCK_TOOLS[tool_name]

    @app.post("/tools/{tool_name}/call")
    async def call_tool(tool_name: str, request: ToolCallRequest):
        """Execute a tool."""
        if tool_name not in MOCK_TOOLS:
            raise HTTPException(status_code=404, detail="Tool not found")
        result = await execute_mock_tool(tool_name, request.arguments)
        return result

    return app


class MockMCPServer:
    """Wrapper for the mock MCP server."""

    def __init__(self, port: int = 8080):
        self.port = port
        self.app = create_mock_mcp_app()

    def get_discovery_response(self) -> dict:
        """Get the discovery response for this server."""
        return {
            "servers": [
                {
                    "id": "mock-server",
                    "name": "Mock MCP Server",
                    "url": f"http://localhost:{self.port}",
                    "description": "Mock server for testing",
                }
            ]
        }
