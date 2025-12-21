"""MCP client for communicating with MCP servers.

Resilience patterns applied:
- Retry with exponential backoff for transient failures
- Circuit breaker to prevent cascade failures to unhealthy servers
- Timeout to bound operation duration
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx

from agentic_chatbot.core.exceptions import MCPError
from agentic_chatbot.core.resilience import (
    mcp_retry,
    mcp_circuit_breaker,
    TransientError,
    RateLimitError,
    BreakerOpen,
    ResilienceConfig,
)
from agentic_chatbot.mcp.models import (
    ToolResult,
    ToolResultStatus,
    ToolContent,
    ToolSchema,
    ToolSummary,
)
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class MCPStreamEvent:
    """Event from MCP streaming response."""

    def __init__(self, event_type: str, data: dict[str, Any]):
        self.type = event_type
        self.data = data

    @property
    def is_progress(self) -> bool:
        return self.type == "progress"

    @property
    def is_content(self) -> bool:
        return self.type == "content"

    @property
    def is_elicitation(self) -> bool:
        return self.type == "elicitation"

    @property
    def is_result(self) -> bool:
        return self.type == "result"

    @property
    def is_error(self) -> bool:
        return self.type == "error"

    @property
    def result(self) -> ToolResult | None:
        """Get result if this is a result event."""
        if not self.is_result:
            return None
        return ToolResult(**self.data.get("result", {}))


class MCPClient:
    """
    Async HTTP client for MCP server communication.

    Features:
    - Automatic retry with exponential backoff for transient errors
    - Streaming support via SSE
    - Connection lifecycle management
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
    ):
        """
        Initialize MCP client.

        Args:
            base_url: Base URL of the MCP server
            timeout: Request timeout in seconds
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @mcp_retry
    @mcp_circuit_breaker
    async def list_tools(self) -> list[ToolSummary]:
        """
        Get list of tools from the server.

        Resilience: Retry with exponential backoff + circuit breaker.

        Returns:
            List of tool summaries
        """
        client = await self._get_client()

        try:
            response = await client.get("/tools")
            response.raise_for_status()
            data = response.json()

            return [
                ToolSummary(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    server_id=data.get("server_id", "unknown"),
                )
                for tool in data.get("tools", [])
            ]
        except httpx.TimeoutException as e:
            raise TransientError(f"Timeout listing tools: {e}") from e
        except httpx.ConnectError as e:
            raise TransientError(f"Connection error: {e}") from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(f"Rate limited: {e}") from e
            if e.response.status_code >= 500:
                raise TransientError(f"Server error: {e}") from e
            raise MCPError(f"Failed to list tools: {e}") from e

    @mcp_retry
    @mcp_circuit_breaker
    async def get_tool_schema(self, tool_name: str) -> ToolSchema:
        """
        Get full schema for a tool.

        Resilience: Retry with exponential backoff + circuit breaker.

        Args:
            tool_name: Name of the tool

        Returns:
            Full tool schema with input schema
        """
        client = await self._get_client()

        try:
            response = await client.get(f"/tools/{tool_name}/schema")
            response.raise_for_status()
            data = response.json()

            return ToolSchema(
                name=data["name"],
                description=data.get("description", ""),
                server_id=data.get("server_id", "unknown"),
                input_schema=data.get("input_schema", {}),
            )
        except httpx.TimeoutException as e:
            raise TransientError(f"Timeout getting tool schema: {e}") from e
        except httpx.ConnectError as e:
            raise TransientError(f"Connection error: {e}") from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(f"Rate limited: {e}") from e
            if e.response.status_code >= 500:
                raise TransientError(f"Server error: {e}") from e
            raise MCPError(f"Failed to get tool schema: {e}", tool_name=tool_name) from e

    @mcp_retry
    @mcp_circuit_breaker
    async def call_tool(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult:
        """
        Call a tool and get the result (non-streaming).

        Resilience: Retry with exponential backoff + circuit breaker.

        Args:
            tool_name: Name of the tool
            params: Tool parameters

        Returns:
            Tool execution result
        """
        client = await self._get_client()
        start_time = time.time()

        try:
            response = await client.post(
                f"/tools/{tool_name}/call",
                json={"params": params},
            )
            response.raise_for_status()
            data = response.json()

            duration_ms = (time.time() - start_time) * 1000

            # Parse contents
            contents = []
            for content_data in data.get("contents", []):
                contents.append(ToolContent(**content_data))

            # Handle simple text response
            if not contents and "result" in data:
                result_data = data["result"]
                if isinstance(result_data, str):
                    contents.append(ToolContent.text(result_data))
                else:
                    contents.append(
                        ToolContent(content_type="application/json", data=result_data)
                    )

            return ToolResult(
                tool_name=tool_name,
                status=ToolResultStatus.SUCCESS,
                contents=contents,
                duration_ms=duration_ms,
                metadata=data.get("metadata", {}),
            )

        except httpx.TimeoutException as e:
            raise TransientError(f"Timeout calling tool: {e}") from e
        except httpx.ConnectError as e:
            raise TransientError(f"Connection error: {e}") from e
        except httpx.HTTPStatusError as e:
            duration_ms = (time.time() - start_time) * 1000
            if e.response.status_code == 429:
                raise RateLimitError(f"Rate limited: {e}") from e
            if e.response.status_code >= 500:
                raise TransientError(f"Server error: {e}") from e
            return ToolResult(
                tool_name=tool_name,
                status=ToolResultStatus.ERROR,
                error=str(e),
                duration_ms=duration_ms,
            )
        except BreakerOpen as e:
            # Circuit breaker is open - server is unhealthy
            duration_ms = (time.time() - start_time) * 1000
            logger.warning(f"Circuit breaker open for tool {tool_name}: {e}")
            return ToolResult(
                tool_name=tool_name,
                status=ToolResultStatus.ERROR,
                error=f"Service unavailable (circuit breaker open): {e}",
                duration_ms=duration_ms,
            )

    @asynccontextmanager
    async def stream_tool_call(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> AsyncIterator[AsyncIterator[MCPStreamEvent]]:
        """
        Call a tool with streaming response.

        Args:
            tool_name: Name of the tool
            params: Tool parameters

        Yields:
            Async iterator of stream events
        """
        client = await self._get_client()

        async def event_generator() -> AsyncIterator[MCPStreamEvent]:
            try:
                async with client.stream(
                    "POST",
                    f"/tools/{tool_name}/stream",
                    json={"params": params},
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue

                        import json

                        data = json.loads(line[6:])
                        yield MCPStreamEvent(
                            event_type=data.get("type", "unknown"),
                            data=data,
                        )

            except httpx.TimeoutException as e:
                yield MCPStreamEvent("error", {"error": f"Timeout: {e}", "error_type": "timeout"})
            except httpx.ConnectError as e:
                yield MCPStreamEvent(
                    "error", {"error": f"Connection: {e}", "error_type": "connection"}
                )
            except Exception as e:
                yield MCPStreamEvent(
                    "error", {"error": str(e), "error_type": "execution"}
                )

        yield event_generator()

    @mcp_retry
    async def health_check(self) -> bool:
        """
        Check if the server is healthy.

        Resilience: Retry with exponential backoff.

        Returns:
            True if healthy, False otherwise
        """
        try:
            client = await self._get_client()
            response = await client.get("/health", timeout=5.0)
            return response.status_code == 200
        except TransientError:
            # Re-raise for retry
            raise
        except Exception:
            return False
