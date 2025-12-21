"""MCP server registry for tool discovery and caching.

Resilience patterns applied:
- Retry with exponential backoff for transient failures
- Circuit breaker to prevent cascade failures to unhealthy servers
"""

import asyncio
from datetime import datetime
from typing import Any

import httpx

from agentic_chatbot.core.exceptions import ToolNotFoundError, ServerNotFoundError
from agentic_chatbot.core.resilience import (
    mcp_retry,
    mcp_circuit_breaker,
    TransientError,
    RateLimitError,
    BreakerOpen,
    ResilienceConfig,
)
from agentic_chatbot.mcp.models import (
    MCPServerInfo,
    ToolSummary,
    ToolSchema,
    MessagingCapabilities,
    OutputDataType,
)
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

    @mcp_retry
    @mcp_circuit_breaker
    async def _refresh(self) -> None:
        """
        Refresh server and tool caches from discovery service.

        Resilience: Retry with exponential backoff + circuit breaker.
        """
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

        except httpx.TimeoutException as e:
            logger.error("Timeout refreshing registry", error=str(e))
            raise TransientError(f"Timeout refreshing registry: {e}") from e
        except httpx.ConnectError as e:
            logger.error("Connection error refreshing registry", error=str(e))
            raise TransientError(f"Connection error: {e}") from e
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error refreshing registry", error=str(e))
            if e.response.status_code == 429:
                raise RateLimitError(f"Rate limited: {e}") from e
            if e.response.status_code >= 500:
                raise TransientError(f"Server error: {e}") from e
            raise
        except BreakerOpen as e:
            logger.warning("Circuit breaker open for registry refresh", error=str(e))
            raise
        except Exception as e:
            logger.error("Failed to refresh registry", error=str(e))
            # Don't update last_refresh so we retry on next access
            raise

    @mcp_retry
    async def _fetch_server_tools(self, server: MCPServerInfo) -> None:
        """
        Fetch tool summaries from a server including messaging capabilities.

        Resilience: Retry with exponential backoff.
        Note: No circuit breaker here as we're iterating over servers
        and one failure shouldn't break the whole refresh.
        """
        try:
            client = await self._get_http_client()
            response = await client.get(f"{server.url}/tools")
            response.raise_for_status()
            data = response.json()

            for tool_data in data.get("tools", []):
                tool_name = tool_data["name"]

                # Parse messaging capabilities if provided
                messaging = self._parse_messaging_capabilities(
                    tool_data.get("messaging", {})
                )

                summary = ToolSummary(
                    name=tool_name,
                    description=tool_data.get("description", ""),
                    server_id=server.id,
                    messaging=messaging,
                )
                self._tool_summaries[tool_name] = summary
                self._tool_to_server[tool_name] = server.id

        except httpx.TimeoutException as e:
            logger.warning(
                "Timeout fetching tools from server",
                server_id=server.id,
                error=str(e),
            )
            raise TransientError(f"Timeout fetching tools: {e}") from e
        except httpx.ConnectError as e:
            logger.warning(
                "Connection error fetching tools from server",
                server_id=server.id,
                error=str(e),
            )
            raise TransientError(f"Connection error: {e}") from e
        except Exception as e:
            logger.warning(
                "Failed to fetch tools from server",
                server_id=server.id,
                error=str(e),
            )

    def _parse_messaging_capabilities(
        self, messaging_data: dict[str, Any]
    ) -> MessagingCapabilities:
        """
        Parse messaging capabilities from tool metadata.

        Args:
            messaging_data: Dictionary with messaging capability fields

        Returns:
            MessagingCapabilities instance
        """
        if not messaging_data:
            return MessagingCapabilities.default()

        # Parse output types
        output_types_raw = messaging_data.get("output_types", ["text"])
        output_types = []
        for ot in output_types_raw:
            try:
                output_types.append(OutputDataType(ot))
            except ValueError:
                # Unknown output type, default to TEXT
                output_types.append(OutputDataType.TEXT)

        if not output_types:
            output_types = [OutputDataType.TEXT]

        return MessagingCapabilities(
            output_types=output_types,
            supports_progress=messaging_data.get("supports_progress", False),
            supports_elicitation=messaging_data.get("supports_elicitation", False),
            supports_direct_response=messaging_data.get("supports_direct_response", False),
            supports_streaming=messaging_data.get("supports_streaming", False),
        )

    @mcp_retry
    @mcp_circuit_breaker
    async def _fetch_tool_schema(self, server_id: str, tool_name: str) -> ToolSchema:
        """
        Fetch full tool schema from server including messaging capabilities.

        Resilience: Retry with exponential backoff + circuit breaker.
        """
        server = self._servers.get(server_id)
        if not server:
            raise ServerNotFoundError(server_id)

        try:
            client = await self._get_http_client()
            response = await client.get(f"{server.url}/tools/{tool_name}/schema")
            response.raise_for_status()
            data = response.json()

            # Parse messaging capabilities if provided
            messaging = self._parse_messaging_capabilities(
                data.get("messaging", {})
            )

            return ToolSchema(
                name=tool_name,
                description=data.get("description", ""),
                server_id=server_id,
                input_schema=data.get("input_schema", {}),
                messaging=messaging,
            )

        except httpx.TimeoutException as e:
            logger.error(
                "Timeout fetching tool schema",
                server_id=server_id,
                tool_name=tool_name,
                error=str(e),
            )
            raise TransientError(f"Timeout fetching schema: {e}") from e
        except httpx.ConnectError as e:
            logger.error(
                "Connection error fetching tool schema",
                server_id=server_id,
                tool_name=tool_name,
                error=str(e),
            )
            raise TransientError(f"Connection error: {e}") from e
        except httpx.HTTPStatusError as e:
            logger.error(
                "HTTP error fetching tool schema",
                server_id=server_id,
                tool_name=tool_name,
                status_code=e.response.status_code,
                error=str(e),
            )
            if e.response.status_code == 429:
                raise RateLimitError(f"Rate limited: {e}") from e
            if e.response.status_code >= 500:
                raise TransientError(f"Server error: {e}") from e
            raise ToolNotFoundError(tool_name) from e
        except BreakerOpen as e:
            logger.warning(
                "Circuit breaker open for schema fetch",
                server_id=server_id,
                tool_name=tool_name,
                error=str(e),
            )
            raise
        except Exception as e:
            logger.error(
                "Failed to fetch tool schema",
                server_id=server_id,
                tool_name=tool_name,
                error=str(e),
            )
            raise ToolNotFoundError(tool_name) from e

    async def get_tool_messaging_capabilities(
        self, tool_name: str
    ) -> MessagingCapabilities:
        """
        Get messaging capabilities for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            MessagingCapabilities instance

        Raises:
            ToolNotFoundError: If tool not found
        """
        summary = await self.get_tool_summary(tool_name)
        return summary.messaging

    async def get_tools_with_capability(
        self,
        supports_progress: bool | None = None,
        supports_elicitation: bool | None = None,
        supports_direct_response: bool | None = None,
        supports_streaming: bool | None = None,
        output_type: OutputDataType | None = None,
    ) -> list[ToolSummary]:
        """
        Get tools that match the specified capability requirements.

        Args:
            supports_progress: Filter by progress support
            supports_elicitation: Filter by elicitation support
            supports_direct_response: Filter by direct response support
            supports_streaming: Filter by streaming support
            output_type: Filter by output type support

        Returns:
            List of ToolSummary instances matching all criteria
        """
        await self._ensure_fresh()
        matching = []

        for summary in self._tool_summaries.values():
            messaging = summary.messaging

            # Check each criterion
            if supports_progress is not None:
                if messaging.supports_progress != supports_progress:
                    continue

            if supports_elicitation is not None:
                if messaging.supports_elicitation != supports_elicitation:
                    continue

            if supports_direct_response is not None:
                if messaging.supports_direct_response != supports_direct_response:
                    continue

            if supports_streaming is not None:
                if messaging.supports_streaming != supports_streaming:
                    continue

            if output_type is not None:
                if output_type not in messaging.output_types:
                    continue

            matching.append(summary)

        return matching

    async def get_widget_capable_tools(self) -> list[ToolSummary]:
        """Get tools that can return widgets (directly to user)."""
        return await self.get_tools_with_capability(
            output_type=OutputDataType.WIDGET,
            supports_direct_response=True,
        )

    async def get_image_capable_tools(self) -> list[ToolSummary]:
        """Get tools that can return images."""
        return await self.get_tools_with_capability(
            output_type=OutputDataType.IMAGE,
        )

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def get_tool_summaries_text(self) -> str:
        """
        Get formatted text of all tool summaries for prompts.

        Includes messaging capabilities information.

        Returns:
            Formatted string of tool summaries
        """
        lines = []
        for name, summary in sorted(self._tool_summaries.items()):
            messaging = summary.messaging
            capabilities = []
            if messaging.supports_progress:
                capabilities.append("progress")
            if messaging.supports_elicitation:
                capabilities.append("elicitation")
            if messaging.supports_direct_response:
                capabilities.append("direct_response")
            if messaging.supports_streaming:
                capabilities.append("streaming")

            output_types = [t.value for t in messaging.output_types]
            cap_str = f" [caps: {', '.join(capabilities)}]" if capabilities else ""
            out_str = f" [outputs: {', '.join(output_types)}]"

            lines.append(f"- {name}: {summary.description}{out_str}{cap_str}")
        return "\n".join(lines) if lines else "No tools available"
