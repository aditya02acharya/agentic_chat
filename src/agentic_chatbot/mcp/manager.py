"""MCP client manager for concurrency control and health tracking."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from agentic_chatbot.mcp.client import MCPClient
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class MCPClientManager:
    """
    Manages MCP client instances per server.

    NOT a connection pool - httpx handles HTTP connection pooling internally.
    This class provides:
    - One httpx client per server (lazy created)
    - Semaphore for concurrency control
    - Health tracking per server
    - Graceful shutdown

    Why not a connection pool?
    - MCP uses HTTP/SSE which already benefits from httpx's built-in connection pooling
    - HTTP/2 multiplexing handles concurrent requests efficiently
    - Adding our own pool on top would be redundant complexity

    Design Pattern: Registry + Factory
    """

    def __init__(
        self,
        max_concurrent_per_server: int = 10,
        timeout_seconds: float = 30.0,
    ):
        """
        Initialize client manager.

        Args:
            max_concurrent_per_server: Max concurrent requests per server
            timeout_seconds: Request timeout
        """
        self._max_concurrent = max_concurrent_per_server
        self._timeout = timeout_seconds
        self._clients: dict[str, MCPClient] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._health: dict[str, bool] = {}
        self._lock = asyncio.Lock()
        self._shutdown = False

    async def _get_or_create_client(self, server_id: str, server_url: str) -> MCPClient:
        """Get existing client or create new one."""
        async with self._lock:
            if server_id not in self._clients:
                logger.debug("Creating MCP client", server_id=server_id, url=server_url)
                self._clients[server_id] = MCPClient(
                    base_url=server_url,
                    timeout=self._timeout,
                )
                self._health[server_id] = True
            return self._clients[server_id]

    def _get_semaphore(self, server_id: str) -> asyncio.Semaphore:
        """Get or create semaphore for server."""
        if server_id not in self._semaphores:
            self._semaphores[server_id] = asyncio.Semaphore(self._max_concurrent)
        return self._semaphores[server_id]

    @asynccontextmanager
    async def acquire(self, server_id: str, server_url: str) -> AsyncIterator[MCPClient]:
        """
        Acquire client with concurrency control.

        Args:
            server_id: Unique server identifier
            server_url: Server base URL (used if client needs to be created)

        Yields:
            MCPClient instance for the server
        """
        if self._shutdown:
            raise RuntimeError("Manager is shutting down")

        semaphore = self._get_semaphore(server_id)
        async with semaphore:
            client = await self._get_or_create_client(server_id, server_url)
            try:
                yield client
                # Mark healthy on success
                self._health[server_id] = True
            except Exception as e:
                # Mark server as unhealthy on error
                logger.warning("MCP client error, marking unhealthy", server_id=server_id, error=str(e))
                self._health[server_id] = False
                raise

    def is_healthy(self, server_id: str) -> bool:
        """Check if server is marked as healthy."""
        return self._health.get(server_id, True)

    def mark_healthy(self, server_id: str) -> None:
        """Mark server as healthy (e.g., after successful call)."""
        self._health[server_id] = True

    def mark_unhealthy(self, server_id: str) -> None:
        """Mark server as unhealthy."""
        self._health[server_id] = False

    async def health_check(self, server_id: str, server_url: str) -> bool:
        """
        Perform health check on a server.

        Args:
            server_id: Server identifier
            server_url: Server URL

        Returns:
            True if healthy
        """
        try:
            client = await self._get_or_create_client(server_id, server_url)
            healthy = await client.health_check()
            self._health[server_id] = healthy
            return healthy
        except Exception as e:
            logger.warning("Health check failed", server_id=server_id, error=str(e))
            self._health[server_id] = False
            return False

    async def shutdown(self) -> None:
        """Graceful shutdown - close all clients."""
        logger.info("Shutting down MCP client manager")
        self._shutdown = True
        async with self._lock:
            for server_id, client in self._clients.items():
                try:
                    await client.close()
                    logger.debug("Closed MCP client", server_id=server_id)
                except Exception as e:
                    logger.warning("Error closing client", server_id=server_id, error=str(e))
            self._clients.clear()
            self._semaphores.clear()
            self._health.clear()

    @property
    def is_shutting_down(self) -> bool:
        """Check if manager is shutting down."""
        return self._shutdown

    def get_all_health_status(self) -> dict[str, bool]:
        """Get health status of all tracked servers."""
        return dict(self._health)
