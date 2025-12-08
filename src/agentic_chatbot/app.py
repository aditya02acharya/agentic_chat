"""Application class with lifecycle management."""

import asyncio
import signal
from typing import Set

from .config.settings import get_settings
from .mcp.client import MCPClient
from .mcp.manager import MCPClientManager
from .mcp.registry import MCPServerRegistry
from .mcp.session import MCPSessionManager
from .utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class Application:
    """
    FastAPI application with graceful shutdown.

    Handles:
    - SIGTERM / SIGINT signals
    - Cleanup of MCP client manager
    - Draining in-flight requests
    - Timeout for cleanup operations
    """

    def __init__(self):
        self.mcp_client_manager: MCPClientManager | None = None
        self.mcp_registry: MCPServerRegistry | None = None
        self.mcp_session_manager: MCPSessionManager | None = None
        self._active_requests: Set[str] = set()
        self._shutdown_event = asyncio.Event()

    async def startup(self) -> None:
        """Initialize resources on startup."""
        settings = get_settings()
        setup_logging(settings.log_level)

        self.mcp_registry = MCPServerRegistry(
            discovery_url=settings.mcp_discovery_url,
            cache_ttl=settings.mcp_cache_ttl_seconds,
        )

        self.mcp_client_manager = MCPClientManager(
            max_concurrent_per_server=settings.mcp_max_concurrent_per_server,
            timeout_seconds=settings.mcp_timeout_seconds,
        )

        self.mcp_session_manager = MCPSessionManager(
            client_manager=self.mcp_client_manager,
            registry=self.mcp_registry,
        )

        try:
            await self.mcp_registry.refresh()
        except Exception as e:
            logger.warning(f"Failed to refresh MCP registry: {e}")

        logger.info("Application started")

    async def shutdown(self, timeout: float = 30.0) -> None:
        """
        Graceful shutdown with timeout.

        1. Stop accepting new requests
        2. Wait for in-flight requests (with timeout)
        3. Close MCP clients
        4. Final cleanup
        """
        logger.info("Shutdown initiated...")
        self._shutdown_event.set()

        if self._active_requests:
            logger.info(f"Waiting for {len(self._active_requests)} requests...")
            try:
                await asyncio.wait_for(
                    self._wait_for_requests(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for requests, forcing shutdown")

        if self.mcp_client_manager:
            await self.mcp_client_manager.shutdown()

        if self.mcp_registry:
            await self.mcp_registry.close()

        logger.info("Shutdown complete")

    async def _wait_for_requests(self) -> None:
        """Wait until all requests complete."""
        while self._active_requests:
            await asyncio.sleep(0.1)

    def track_request(self, request_id: str) -> None:
        """Track active request."""
        self._active_requests.add(request_id)

    def untrack_request(self, request_id: str) -> None:
        """Remove request from tracking."""
        self._active_requests.discard(request_id)

    @property
    def is_shutting_down(self) -> bool:
        return self._shutdown_event.is_set()
