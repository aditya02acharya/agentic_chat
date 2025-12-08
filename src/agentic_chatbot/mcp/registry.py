"""MCP server registry with discovery and caching."""

import asyncio
import time
from typing import Any

from .client import MCPClient
from .models import ServerInfo, ToolSummary, ToolSchema
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MCPServerRegistry:
    """
    Registry for MCP servers and tools.

    Features:
    - Server discovery with caching
    - Tool name to server mapping
    - Lazy schema loading
    - TTL-based cache refresh
    """

    def __init__(self, discovery_url: str, cache_ttl: int = 300):
        self.discovery_url = discovery_url
        self.cache_ttl = cache_ttl
        self._client = MCPClient()
        self._servers: dict[str, ServerInfo] = {}
        self._tool_summaries: dict[str, ToolSummary] = {}
        self._tool_schemas: dict[str, ToolSchema] = {}
        self._last_refresh: float = 0
        self._lock = asyncio.Lock()

    @property
    def servers(self) -> list[ServerInfo]:
        """Get list of known servers."""
        return list(self._servers.values())

    @property
    def tools(self) -> list[ToolSummary]:
        """Get list of known tools."""
        return list(self._tool_summaries.values())

    def get_server(self, server_id: str) -> ServerInfo | None:
        """Get server by ID."""
        return self._servers.get(server_id)

    def get_tool_summary(self, tool_name: str) -> ToolSummary | None:
        """Get tool summary by name."""
        return self._tool_summaries.get(tool_name)

    def get_server_for_tool(self, tool_name: str) -> ServerInfo | None:
        """Get server that hosts a tool."""
        summary = self._tool_summaries.get(tool_name)
        if summary:
            return self._servers.get(summary.server_id)
        return None

    async def refresh(self) -> None:
        """Refresh server and tool lists from discovery."""
        async with self._lock:
            now = time.time()
            if now - self._last_refresh < self.cache_ttl:
                return

            logger.info("Refreshing MCP server registry")

            servers = await self._client.discover_servers(self.discovery_url)
            self._servers = {s.id: s for s in servers}

            self._tool_summaries.clear()
            for server in servers:
                tools = await self._client.list_tools(server)
                for tool in tools:
                    self._tool_summaries[tool.name] = tool

            self._last_refresh = now
            logger.info(
                f"Registry refreshed: {len(self._servers)} servers, "
                f"{len(self._tool_summaries)} tools"
            )

    async def get_tool_schema(self, tool_name: str) -> ToolSchema | None:
        """Get full schema for a tool (lazy loaded)."""
        if tool_name in self._tool_schemas:
            return self._tool_schemas[tool_name]

        summary = self._tool_summaries.get(tool_name)
        if not summary:
            return None

        server = self._servers.get(summary.server_id)
        if not server:
            return None

        schema = await self._client.get_tool_schema(server, tool_name)
        if schema:
            self._tool_schemas[tool_name] = schema
        return schema

    async def close(self) -> None:
        """Close the registry client."""
        await self._client.close()
