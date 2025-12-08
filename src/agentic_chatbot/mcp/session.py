"""MCP session management."""

from typing import Any, Callable, Awaitable

from .manager import MCPClientManager
from .registry import MCPServerRegistry
from .models import ToolResult
from .callbacks import MCPProgressCallback, MCPContentCallback, MCPErrorCallback
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MCPSession:
    """
    Session for executing MCP tool calls.

    Features:
    - Bound to a specific request
    - Callback support for progress/content/error
    - Context manager for cleanup
    """

    def __init__(
        self,
        client_manager: MCPClientManager,
        registry: MCPServerRegistry,
        on_progress: MCPProgressCallback | None = None,
        on_content: MCPContentCallback | None = None,
        on_error: MCPErrorCallback | None = None,
    ):
        self._manager = client_manager
        self._registry = registry
        self._on_progress = on_progress
        self._on_content = on_content
        self._on_error = on_error
        self._closed = False

    async def call_tool(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult:
        """Execute a tool by name."""
        if self._closed:
            raise RuntimeError("Session is closed")

        server = self._registry.get_server_for_tool(tool_name)
        if not server:
            return ToolResult(
                tool_name=tool_name,
                server_id="unknown",
                success=False,
                error=f"No server found for tool: {tool_name}",
            )

        result = await self._manager.call_tool(server, tool_name, params)

        if not result.success and self._on_error:
            await self._on_error(
                server.id,
                tool_name,
                result.error or "Unknown error",
                recoverable=True,
            )

        if result.success and result.content and self._on_content:
            for content in result.content:
                await self._on_content(
                    server.id,
                    tool_name,
                    content.content_type,
                    content.data,
                )

        return result

    async def close(self) -> None:
        """Close the session."""
        self._closed = True

    async def __aenter__(self) -> "MCPSession":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


class MCPSessionManager:
    """
    Factory for creating MCP sessions.

    Features:
    - Creates sessions per request
    - Shares client manager and registry
    """

    def __init__(
        self,
        client_manager: MCPClientManager,
        registry: MCPServerRegistry,
    ):
        self._manager = client_manager
        self._registry = registry

    def create_session(
        self,
        on_progress: MCPProgressCallback | None = None,
        on_content: MCPContentCallback | None = None,
        on_error: MCPErrorCallback | None = None,
    ) -> MCPSession:
        """Create a new MCP session."""
        return MCPSession(
            client_manager=self._manager,
            registry=self._registry,
            on_progress=on_progress,
            on_content=on_content,
            on_error=on_error,
        )
