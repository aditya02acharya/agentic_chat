"""MCP session management for tool execution."""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from agentic_chatbot.core.exceptions import MCPError
from agentic_chatbot.mcp.callbacks import MCPCallbacks
from agentic_chatbot.mcp.client import MCPClient, MCPStreamEvent
from agentic_chatbot.mcp.models import (
    ToolCall,
    ToolResult,
    ToolResultStatus,
    ToolContent,
    ElicitationRequest,
)
from agentic_chatbot.mcp.manager import MCPClientManager
from agentic_chatbot.mcp.registry import MCPServerRegistry
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class MCPSession:
    """
    Active MCP session for tool execution.

    Design Pattern: Context Manager for resource cleanup

    Features:
    - Bound to tool call(s)
    - Progress callback → Events
    - Elicitation callback → User
    - Content callback → Stream
    - Auto-cleanup on exit/error
    """

    def __init__(
        self,
        client: MCPClient,
        server_id: str,
        callbacks: MCPCallbacks,
    ):
        """
        Initialize MCP session.

        Args:
            client: MCP client instance
            server_id: Server identifier
            callbacks: Callback handlers
        """
        self._client = client
        self._server_id = server_id
        self._callbacks = callbacks
        self._active_streams: list[Any] = []

    async def call_tool(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult:
        """
        Execute single tool with streaming support.

        Handles progress, elicitation, and content callbacks automatically.

        Args:
            tool_name: Name of the tool
            params: Tool parameters

        Returns:
            Tool execution result
        """
        logger.debug("Calling tool", server_id=self._server_id, tool_name=tool_name)
        start_time = time.time()

        try:
            async with self._client.stream_tool_call(tool_name, params) as stream:
                async for event in stream:
                    await self._handle_event(tool_name, event)
                    if event.is_result:
                        result = event.result
                        if result:
                            result.duration_ms = (time.time() - start_time) * 1000
                            return result
                    if event.is_error:
                        duration_ms = (time.time() - start_time) * 1000
                        return ToolResult(
                            tool_name=tool_name,
                            status=ToolResultStatus.ERROR,
                            error=event.data.get("error", "Unknown error"),
                            duration_ms=duration_ms,
                        )

            # If no result event, fall back to non-streaming call
            logger.debug("No streaming result, falling back to sync call", tool_name=tool_name)
            return await self._client.call_tool(tool_name, params)

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error("Tool call failed", tool_name=tool_name, error=str(e))
            if self._callbacks.on_error:
                await self._callbacks.on_error(
                    self._server_id, tool_name, str(e), "execution"
                )
            return ToolResult(
                tool_name=tool_name,
                status=ToolResultStatus.ERROR,
                error=str(e),
                duration_ms=duration_ms,
            )

    async def call_tools_parallel(
        self,
        calls: list[ToolCall],
    ) -> list[ToolResult]:
        """
        Execute multiple tools concurrently.

        Args:
            calls: List of tool calls

        Returns:
            List of results (in same order as calls)
        """
        results = await asyncio.gather(
            *[self.call_tool(c.tool_name, c.params) for c in calls],
            return_exceptions=True,
        )

        # Convert exceptions to error results
        final_results = []
        for call, result in zip(calls, results):
            if isinstance(result, Exception):
                final_results.append(
                    ToolResult(
                        tool_name=call.tool_name,
                        status=ToolResultStatus.ERROR,
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)

        return final_results

    async def _handle_event(self, tool_name: str, event: MCPStreamEvent) -> None:
        """Handle streaming event from MCP server."""
        if event.is_progress and self._callbacks.on_progress:
            await self._callbacks.on_progress(
                self._server_id,
                tool_name,
                event.data.get("progress", 0.0),
                event.data.get("message", ""),
            )

        elif event.is_content and self._callbacks.on_content:
            await self._callbacks.on_content(
                self._server_id,
                tool_name,
                event.data.get("data"),
                event.data.get("content_type", "text/plain"),
            )

        elif event.is_elicitation and self._callbacks.on_elicitation:
            request = ElicitationRequest(
                request_id=event.data.get("request_id", ""),
                prompt=event.data.get("prompt", ""),
                input_type=event.data.get("input_type", "text"),
                options=event.data.get("options"),
                default=event.data.get("default"),
                timeout_seconds=event.data.get("timeout_seconds", 60.0),
            )
            # Note: Response handling would need to be implemented
            # based on how the MCP server expects the response
            await self._callbacks.on_elicitation(self._server_id, tool_name, request)

        elif event.is_error and self._callbacks.on_error:
            await self._callbacks.on_error(
                self._server_id,
                tool_name,
                event.data.get("error", "Unknown error"),
                event.data.get("error_type", "execution"),
            )

    async def close(self) -> None:
        """Cancel all active streams and cleanup."""
        for stream in self._active_streams:
            try:
                await stream.aclose()
            except Exception:
                pass
        self._active_streams.clear()


class MCPSessionManager:
    """
    Manages MCP session lifecycle with automatic cleanup.

    Design Pattern: Factory + Context Manager
    """

    def __init__(
        self,
        client_manager: MCPClientManager,
        registry: MCPServerRegistry,
    ):
        """
        Initialize session manager.

        Args:
            client_manager: Client manager for acquiring clients
            registry: Server registry for looking up servers
        """
        self._client_manager = client_manager
        self._registry = registry

    @asynccontextmanager
    async def session(
        self,
        server_id: str,
        callbacks: MCPCallbacks,
    ) -> AsyncIterator[MCPSession]:
        """
        Create session with guaranteed cleanup.

        Usage:
            async with manager.session("rag_server", callbacks) as session:
                result = await session.call_tool("rag_search", {"query": "..."})
            # Session automatically cleaned up here

        Args:
            server_id: ID of the server to connect to
            callbacks: Callback handlers

        Yields:
            Active MCP session
        """
        server_info = await self._registry.get_server(server_id)
        async with self._client_manager.acquire(server_id, server_info.url) as client:
            session = MCPSession(client, server_id, callbacks)
            try:
                yield session
            finally:
                await session.close()

    @asynccontextmanager
    async def session_for_tool(
        self,
        tool_name: str,
        callbacks: MCPCallbacks,
    ) -> AsyncIterator[MCPSession]:
        """
        Create session for a specific tool (looks up server automatically).

        Args:
            tool_name: Name of the tool
            callbacks: Callback handlers

        Yields:
            Active MCP session
        """
        server_id = await self._registry.get_server_for_tool(tool_name)
        async with self.session(server_id, callbacks) as session:
            yield session
