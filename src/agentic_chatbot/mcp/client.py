"""MCP client for communicating with MCP servers."""

import time
from typing import Any

import httpx

from .models import ServerInfo, ToolSummary, ToolSchema, ToolResult, ToolContent
from ..core.exceptions import MCPError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MCPClient:
    """
    Async HTTP client for MCP servers.

    Features:
    - Connection pooling via httpx
    - Timeout handling
    - Error normalization
    """

    def __init__(self, timeout_seconds: int = 30):
        self.timeout = httpx.Timeout(timeout_seconds)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def discover_servers(self, discovery_url: str) -> list[ServerInfo]:
        """Fetch list of available MCP servers."""
        try:
            client = await self._get_client()
            response = await client.get(discovery_url)
            response.raise_for_status()
            data = response.json()

            servers = []
            for item in data.get("servers", []):
                servers.append(ServerInfo(
                    id=item["id"],
                    name=item["name"],
                    url=item["url"],
                    description=item.get("description"),
                ))
            return servers

        except httpx.RequestError as e:
            logger.warning(f"MCP discovery failed: {e}")
            return []

    async def list_tools(self, server: ServerInfo) -> list[ToolSummary]:
        """Get tool summaries from a server."""
        try:
            client = await self._get_client()
            response = await client.get(f"{server.url}/tools")
            response.raise_for_status()
            data = response.json()

            tools = []
            for item in data.get("tools", []):
                tools.append(ToolSummary(
                    name=item["name"],
                    description=item.get("description", ""),
                    server_id=server.id,
                ))
            return tools

        except httpx.RequestError as e:
            logger.warning(f"Failed to list tools from {server.id}: {e}")
            return []

    async def get_tool_schema(
        self, server: ServerInfo, tool_name: str
    ) -> ToolSchema | None:
        """Get full schema for a specific tool."""
        try:
            client = await self._get_client()
            response = await client.get(f"{server.url}/tools/{tool_name}")
            response.raise_for_status()
            data = response.json()

            return ToolSchema(
                name=data["name"],
                description=data.get("description", ""),
                server_id=server.id,
                input_schema=data.get("inputSchema", {}),
            )

        except httpx.RequestError as e:
            logger.warning(f"Failed to get schema for {tool_name}: {e}")
            return None

    async def call_tool(
        self,
        server: ServerInfo,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult:
        """Execute a tool on the server."""
        start_time = time.time()

        try:
            client = await self._get_client()
            response = await client.post(
                f"{server.url}/tools/{tool_name}/call",
                json={"arguments": params},
            )
            response.raise_for_status()
            data = response.json()

            duration_ms = (time.time() - start_time) * 1000

            content = []
            for item in data.get("content", []):
                content.append(ToolContent(
                    content_type=item.get("type", "text/plain"),
                    data=item.get("text") or item.get("data"),
                    is_error=item.get("isError", False),
                ))

            return ToolResult(
                tool_name=tool_name,
                server_id=server.id,
                success=not data.get("isError", False),
                content=content,
                duration_ms=duration_ms,
            )

        except httpx.RequestError as e:
            duration_ms = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=tool_name,
                server_id=server.id,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )

        except httpx.HTTPStatusError as e:
            duration_ms = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=tool_name,
                server_id=server.id,
                success=False,
                error=f"HTTP {e.response.status_code}: {e.response.text[:200]}",
                duration_ms=duration_ms,
            )
