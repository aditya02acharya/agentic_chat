"""MCP callback protocols and types."""

from dataclasses import dataclass
from typing import Protocol, Any

from agentic_chatbot.mcp.models import ElicitationRequest, ElicitationResponse


class MCPProgressCallback(Protocol):
    """Called when MCP tool reports progress."""

    async def __call__(
        self,
        server_id: str,
        tool_name: str,
        progress: float,  # 0.0 - 1.0
        message: str,
    ) -> None:
        """
        Handle progress update.

        Args:
            server_id: ID of the MCP server
            tool_name: Name of the tool reporting progress
            progress: Progress value (0.0 to 1.0)
            message: Progress message
        """
        ...


class MCPElicitationCallback(Protocol):
    """
    Called when MCP tool needs user input.

    MUST return a response - tool execution blocks until response received.
    Implement timeout in the callback if needed.
    """

    async def __call__(
        self,
        server_id: str,
        tool_name: str,
        request: ElicitationRequest,
    ) -> ElicitationResponse:
        """
        Handle elicitation request.

        Args:
            server_id: ID of the MCP server
            tool_name: Name of the tool requesting input
            request: Elicitation request details

        Returns:
            User's response to the elicitation
        """
        ...


class MCPContentCallback(Protocol):
    """Called when MCP tool streams content (images, rich data, etc)."""

    async def __call__(
        self,
        server_id: str,
        tool_name: str,
        content: Any,
        content_type: str,  # MIME type
    ) -> None:
        """
        Handle streamed content.

        Args:
            server_id: ID of the MCP server
            tool_name: Name of the tool streaming content
            content: Content data
            content_type: MIME type of the content
        """
        ...


class MCPErrorCallback(Protocol):
    """Called when MCP tool encounters an error."""

    async def __call__(
        self,
        server_id: str,
        tool_name: str,
        error: str,
        error_type: str,  # "timeout" | "connection" | "execution" | "validation"
    ) -> None:
        """
        Handle error from MCP tool.

        Args:
            server_id: ID of the MCP server
            tool_name: Name of the tool that errored
            error: Error message
            error_type: Type of error
        """
        ...


@dataclass
class MCPCallbacks:
    """Container for MCP callbacks - passed to session creation."""

    on_progress: MCPProgressCallback | None = None
    on_elicitation: MCPElicitationCallback | None = None
    on_content: MCPContentCallback | None = None
    on_error: MCPErrorCallback | None = None
