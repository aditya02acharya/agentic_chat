"""MCP client manager with concurrency control."""

import asyncio
from typing import Any

from .client import MCPClient
from .models import ServerInfo, ToolResult
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MCPClientManager:
    """
    Manages MCP client connections with concurrency control.

    Features:
    - Semaphore-based concurrency limiting per server
    - Health tracking
    - Graceful shutdown
    """

    def __init__(
        self,
        max_concurrent_per_server: int = 10,
        timeout_seconds: int = 30,
    ):
        self.max_concurrent = max_concurrent_per_server
        self.timeout_seconds = timeout_seconds
        self._client = MCPClient(timeout_seconds=timeout_seconds)
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._health: dict[str, bool] = {}

    def _get_semaphore(self, server_id: str) -> asyncio.Semaphore:
        """Get or create semaphore for server."""
        if server_id not in self._semaphores:
            self._semaphores[server_id] = asyncio.Semaphore(self.max_concurrent)
        return self._semaphores[server_id]

    async def call_tool(
        self,
        server: ServerInfo,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult:
        """Call tool with concurrency limiting."""
        semaphore = self._get_semaphore(server.id)

        async with semaphore:
            result = await self._client.call_tool(server, tool_name, params)

            self._health[server.id] = result.success
            return result

    def is_healthy(self, server_id: str) -> bool:
        """Check if server is considered healthy."""
        return self._health.get(server_id, True)

    async def shutdown(self) -> None:
        """Gracefully shutdown all clients."""
        await self._client.close()
        self._semaphores.clear()
        self._health.clear()
        logger.info("MCPClientManager shutdown complete")
