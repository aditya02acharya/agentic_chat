"""MCP callback protocols and types."""

from typing import Protocol, Any


class MCPProgressCallback(Protocol):
    """Called when MCP tool reports progress."""

    async def __call__(
        self,
        server_id: str,
        tool_name: str,
        progress: float,
        message: str,
    ) -> None:
        ...


class MCPContentCallback(Protocol):
    """Called when MCP tool streams content."""

    async def __call__(
        self,
        server_id: str,
        tool_name: str,
        content_type: str,
        data: Any,
    ) -> None:
        ...


class MCPErrorCallback(Protocol):
    """Called when MCP tool encounters an error."""

    async def __call__(
        self,
        server_id: str,
        tool_name: str,
        error: str,
        recoverable: bool,
    ) -> None:
        ...


class MCPElicitationCallback(Protocol):
    """Called when MCP tool needs user input."""

    async def __call__(
        self,
        server_id: str,
        tool_name: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        ...
