"""MCP server registry for tool discovery and caching."""

import asyncio
from datetime import datetime
from typing import Any

import httpx

from agentic_chatbot.core.exceptions import ToolNotFoundError, ServerNotFoundError
from agentic_chatbot.mcp.models import MCPServerInfo, ToolSummary, ToolSchema
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class MCPServerRegistry:
    """
    Registry for MCP servers and their tools.

    Design Pattern: Registry + Proxy (lazy loading)

    Responsibilities:
    - Discover and cache available MCP servers
    - Map tool names to server IDs
    - Track server health status
    - Refresh periodically (5 min TTL)
    """

    def __init__(
        self,
        discovery_url: str,
        cache_ttl: float = 300.0,
    ):
        """
        Initialize server registry.

        Args:
            discovery_url: URL of MCP discovery service
            cache_ttl: Cache time-to-live in seconds (default 5 min)
        """
        self._discovery_url = discovery_url
        self._cache_ttl = cache_ttl
        self._servers: dict[str, MCPServerInfo] = {}
        self._tool_to_server: dict[str, str] = {}
        self._tool_summaries: dict[str, ToolSummary] = {}
        self._tool_schemas: dict[str, ToolSchema] = {}  # Lazy loaded
        self._last_refresh: datetime | None = None
        self._lock = asyncio.Lock()
        self._http_client: httpx.AsyncClient | None = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def get_all_tool_summaries(self) -> list[ToolSummary]:
        """
        Get summaries of all tools (for Supervisor context).

        Returns:
            List of tool summaries
        """
        await self._ensure_fresh()
        return list(self._tool_summaries.values())

    async def get_server(self, server_id: str) -> MCPServerInfo:
        """
        Get server information by ID.

        Args:
            server_id: Server identifier

        Returns:
            Server information

        Raises:
            ServerNotFoundError: If server not found
        """
        await self._ensure_fresh()
        if server_id not in self._servers:
            raise ServerNotFoundError(server_id)
        return self._servers[server_id]

    async def get_server_for_tool(self, tool_name: str) -> str:
        """
        Look up which server hosts a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Server ID

        Raises:
            ToolNotFoundError: If tool not found
        """
        await self._ensure_fresh()
        if tool_name not in self._tool_to_server:
            raise ToolNotFoundError(tool_name)
        return self._tool_to_server[tool_name]

    async def get_tool_summary(self, tool_name: str) -> ToolSummary:
        """
        Get tool summary by name.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool summary

        Raises:
            ToolNotFoundError: If tool not found
        """
        await self._ensure_fresh()
        if tool_name not in self._tool_summaries:
            raise ToolNotFoundError(tool_name)
        return self._tool_summaries[tool_name]

    async def get_tool_schema(self, tool_name: str) -> ToolSchema:
        """
        Get full tool schema (lazy loaded, cached).

        Args:
            tool_name: Name of the tool

        Returns:
            Full tool schema

        Raises:
            ToolNotFoundError: If tool not found
        """
        if tool_name not in self._tool_schemas:
            server_id = await self.get_server_for_tool(tool_name)
            schema = await self._fetch_tool_schema(server_id, tool_name)
            self._tool_schemas[tool_name] = schema
        return self._tool_schemas[tool_name]

    async def refresh(self) -> None:
        """Force refresh of server and tool caches."""
        async with self._lock:
            await self._refresh()

    async def _ensure_fresh(self) -> None:
        """Refresh cache if stale."""
        async with self._lock:
            if self._should_refresh():
                await self._refresh()

    def _should_refresh(self) -> bool:
        """Check if cache needs refresh."""
        if self._last_refresh is None:
            return True
        age = (datetime.utcnow() - self._last_refresh).total_seconds()
        return age > self._cache_ttl

    async def _refresh(self) -> None:
        """Refresh server and tool caches from discovery service."""
        logger.info("Refreshing MCP server registry", discovery_url=self._discovery_url)

        try:
            client = await self._get_http_client()
            response = await client.get(self._discovery_url)
            response.raise_for_status()
            data = response.json()

            # Clear existing caches
            self._servers.clear()
            self._tool_to_server.clear()
            self._tool_summaries.clear()
            # Don't clear tool_schemas - they're expensive to fetch

            # Parse servers
            for server_data in data.get("servers", []):
                server = MCPServerInfo(
                    id=server_data["id"],
                    name=server_data.get("name", server_data["id"]),
                    url=server_data["url"],
                    description=server_data.get("description", ""),
                    version=server_data.get("version", "1.0.0"),
                    tools=server_data.get("tools", []),
                    healthy=server_data.get("healthy", True),
                )
                self._servers[server.id] = server

                # Fetch tool summaries for this server
                await self._fetch_server_tools(server)

            self._last_refresh = datetime.utcnow()
            logger.info(
                "Registry refreshed",
                servers=len(self._servers),
                tools=len(self._tool_summaries),
            )

        except Exception as e:
            logger.error("Failed to refresh registry", error=str(e))
            # Don't update last_refresh so we retry on next access
            raise

    async def _fetch_server_tools(self, server: MCPServerInfo) -> None:
        """Fetch tool summaries from a server."""
        try:
            client = await self._get_http_client()
            response = await client.get(f"{server.url}/tools")
            response.raise_for_status()
            data = response.json()

            for tool_data in data.get("tools", []):
                tool_name = tool_data["name"]
                summary = ToolSummary(
                    name=tool_name,
                    description=tool_data.get("description", ""),
                    server_id=server.id,
                )
                self._tool_summaries[tool_name] = summary
                self._tool_to_server[tool_name] = server.id

        except Exception as e:
            logger.warning(
                "Failed to fetch tools from server",
                server_id=server.id,
                error=str(e),
            )

    async def _fetch_tool_schema(self, server_id: str, tool_name: str) -> ToolSchema:
        """Fetch full tool schema from server."""
        server = self._servers.get(server_id)
        if not server:
            raise ServerNotFoundError(server_id)

        try:
            client = await self._get_http_client()
            response = await client.get(f"{server.url}/tools/{tool_name}/schema")
            response.raise_for_status()
            data = response.json()

            return ToolSchema(
                name=tool_name,
                description=data.get("description", ""),
                server_id=server_id,
                input_schema=data.get("input_schema", {}),
            )

        except Exception as e:
            logger.error(
                "Failed to fetch tool schema",
                server_id=server_id,
                tool_name=tool_name,
                error=str(e),
            )
            raise ToolNotFoundError(tool_name) from e

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def get_tool_summaries_text(self) -> str:
        """
        Get formatted text of all tool summaries for prompts.

        Returns:
            Formatted string of tool summaries
        """
        lines = []
        for name, summary in sorted(self._tool_summaries.items()):
            lines.append(f"- {name}: {summary.description}")
        return "\n".join(lines) if lines else "No tools available"
