"""Application class with startup/shutdown lifecycle."""

import asyncio
import signal
from typing import Set

from agentic_chatbot.config.settings import get_settings
from agentic_chatbot.mcp.manager import MCPClientManager
from agentic_chatbot.mcp.registry import MCPServerRegistry
from agentic_chatbot.utils.logging import get_logger, configure_logging
from agentic_chatbot.utils.observability import configure_observability, flush_observability
from agentic_chatbot.api.rate_limit import get_rate_limiter


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
        """Initialize application."""
        self.mcp_client_manager: MCPClientManager | None = None
        self.mcp_server_registry: MCPServerRegistry | None = None
        self.document_service = None
        self._active_requests: Set[str] = set()
        self._shutdown_event = asyncio.Event()
        self._is_shutting_down = False

    async def startup(self) -> None:
        """Initialize resources on startup."""
        settings = get_settings()

        # Configure logging
        configure_logging(settings.log_level)

        # Configure observability (Langfuse)
        configure_observability()

        logger.info("Starting application...")

        # Initialize MCP components
        self.mcp_server_registry = MCPServerRegistry(
            discovery_url=settings.mcp_discovery_url,
            cache_ttl=settings.mcp_cache_ttl_seconds,
        )
        self.mcp_client_manager = MCPClientManager(
            max_concurrent_per_server=settings.mcp_max_concurrent_per_server,
            timeout_seconds=settings.mcp_timeout_seconds,
        )

        # Try to warm up registry (non-fatal if fails)
        try:
            await self.mcp_server_registry.refresh()
            logger.info("MCP registry initialized")
        except Exception as e:
            logger.warning(f"MCP registry warmup failed: {e}")

        # Initialize document service
        try:
            from agentic_chatbot.documents.storage import LocalDocumentStorage
            from agentic_chatbot.documents.service import DocumentService
            from agentic_chatbot.utils.llm import LLMClient

            storage = LocalDocumentStorage(
                base_path=settings.document_storage_path
                if hasattr(settings, "document_storage_path")
                else "./storage/documents"
            )
            llm_client = LLMClient()
            self.document_service = DocumentService(
                storage=storage,
                llm_client=llm_client,
            )
            logger.info("Document service initialized")
        except Exception as e:
            logger.warning(f"Document service initialization failed: {e}")
            self.document_service = None

        # Start rate limiter cleanup task
        await get_rate_limiter().start_cleanup_task()

        logger.info("Application started")

    async def shutdown(self, timeout: float = 30.0) -> None:
        """
        Graceful shutdown with timeout.

        Args:
            timeout: Maximum time to wait for requests to complete
        """
        from agentic_chatbot.api.routes import cleanup_background_tasks

        logger.info("Shutdown initiated...")
        self._is_shutting_down = True
        self._shutdown_event.set()

        # Wait for in-flight requests
        if self._active_requests:
            logger.info(f"Waiting for {len(self._active_requests)} requests...")
            try:
                await asyncio.wait_for(
                    self._wait_for_requests(),
                    timeout=timeout / 2,
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for requests, continuing shutdown")

        # Wait for background tasks (document processing)
        await cleanup_background_tasks(timeout=timeout / 4)

        # Stop rate limiter cleanup task
        await get_rate_limiter().stop_cleanup_task()

        # Close MCP client manager
        if self.mcp_client_manager:
            await self.mcp_client_manager.shutdown()

        # Close MCP registry
        if self.mcp_server_registry:
            await self.mcp_server_registry.close()

        # Flush observability traces
        flush_observability()

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
        """Check if application is shutting down."""
        return self._is_shutting_down

    @property
    def active_requests(self) -> Set[str]:
        """Get set of active request IDs."""
        return self._active_requests


# Global application instance
app_instance = Application()


def setup_signal_handlers(app: Application) -> None:
    """
    Setup signal handlers for graceful shutdown.

    Args:
        app: Application instance
    """

    async def handle_signal(sig: signal.Signals) -> None:
        logger.info(f"Received signal {sig.name}")
        await app.shutdown()

    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(handle_signal(s)),
            )
    except NotImplementedError:
        # Windows doesn't support add_signal_handler
        logger.warning("Signal handlers not supported on this platform")
