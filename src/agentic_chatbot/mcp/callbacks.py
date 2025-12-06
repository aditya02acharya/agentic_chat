"""MCP callback protocols, implementations, and elicitation management."""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Protocol, Any

from agentic_chatbot.events.emitter import EventEmitter
from agentic_chatbot.events.models import (
    MCPProgressEvent,
    MCPContentEvent,
    MCPElicitationRequestEvent,
    MCPErrorEvent,
)
from agentic_chatbot.mcp.models import ElicitationRequest, ElicitationResponse


# =============================================================================
# CALLBACK PROTOCOLS
# =============================================================================


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


# =============================================================================
# ELICITATION MANAGER
# =============================================================================


@dataclass
class PendingElicitation:
    """Tracks a pending elicitation request waiting for user response."""

    elicitation_id: str
    server_id: str
    tool_name: str
    request: ElicitationRequest
    future: asyncio.Future[ElicitationResponse]
    created_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())


class ElicitationManager:
    """
    Manages pending elicitation requests and user responses.

    This class bridges the gap between MCP tools requesting user input
    and the API receiving user responses. It:
    1. Tracks pending elicitation requests
    2. Emits events for the UI to display
    3. Waits for and routes user responses back to the tool

    Thread-safe for concurrent request handling.
    """

    def __init__(self) -> None:
        self._pending: dict[str, PendingElicitation] = {}
        self._lock = asyncio.Lock()

    async def create_elicitation(
        self,
        server_id: str,
        tool_name: str,
        request: ElicitationRequest,
    ) -> PendingElicitation:
        """
        Create a new pending elicitation request.

        Args:
            server_id: MCP server ID
            tool_name: Name of the tool requesting input
            request: The elicitation request details

        Returns:
            PendingElicitation with a Future to await
        """
        async with self._lock:
            elicitation_id = request.request_id or str(uuid.uuid4())
            loop = asyncio.get_event_loop()
            future: asyncio.Future[ElicitationResponse] = loop.create_future()

            pending = PendingElicitation(
                elicitation_id=elicitation_id,
                server_id=server_id,
                tool_name=tool_name,
                request=request,
                future=future,
            )
            self._pending[elicitation_id] = pending
            return pending

    async def submit_response(
        self,
        elicitation_id: str,
        value: Any,
        cancelled: bool = False,
    ) -> bool:
        """
        Submit a user response to a pending elicitation.

        Args:
            elicitation_id: ID of the elicitation request
            value: User's response value
            cancelled: Whether user cancelled

        Returns:
            True if response was accepted, False if elicitation not found
        """
        async with self._lock:
            pending = self._pending.pop(elicitation_id, None)
            if pending is None:
                return False

            response = ElicitationResponse(
                request_id=elicitation_id,
                value=value,
                cancelled=cancelled,
            )
            pending.future.set_result(response)
            return True

    async def cancel_elicitation(self, elicitation_id: str) -> bool:
        """
        Cancel a pending elicitation (e.g., on timeout or user abort).

        Args:
            elicitation_id: ID of the elicitation to cancel

        Returns:
            True if cancelled, False if not found
        """
        return await self.submit_response(elicitation_id, None, cancelled=True)

    async def wait_for_response(
        self,
        elicitation_id: str,
        timeout: float | None = None,
    ) -> ElicitationResponse:
        """
        Wait for user response to an elicitation.

        Args:
            elicitation_id: ID of the elicitation
            timeout: Optional timeout in seconds

        Returns:
            User's response

        Raises:
            asyncio.TimeoutError: If timeout exceeded
            KeyError: If elicitation not found
        """
        async with self._lock:
            pending = self._pending.get(elicitation_id)
            if pending is None:
                raise KeyError(f"Elicitation {elicitation_id} not found")
            future = pending.future

        if timeout:
            return await asyncio.wait_for(future, timeout)
        return await future

    def get_pending(self, elicitation_id: str) -> PendingElicitation | None:
        """Get a pending elicitation by ID."""
        return self._pending.get(elicitation_id)

    def get_all_pending(self) -> list[PendingElicitation]:
        """Get all pending elicitations."""
        return list(self._pending.values())

    async def cleanup_expired(self, max_age_seconds: float = 300) -> int:
        """
        Clean up expired elicitations.

        Args:
            max_age_seconds: Maximum age before expiration

        Returns:
            Number of cleaned up elicitations
        """
        async with self._lock:
            now = asyncio.get_event_loop().time()
            expired = [
                eid
                for eid, pending in self._pending.items()
                if now - pending.created_at > max_age_seconds
            ]
            for eid in expired:
                pending = self._pending.pop(eid)
                if not pending.future.done():
                    pending.future.cancel()
            return len(expired)


# =============================================================================
# CONCRETE CALLBACK IMPLEMENTATIONS
# =============================================================================


class MCPProgressHandler:
    """
    Concrete progress callback that emits events to EventEmitter.

    Transforms MCP progress updates into SSE events for the UI.
    """

    def __init__(self, emitter: EventEmitter, request_id: str | None = None):
        self._emitter = emitter
        self._request_id = request_id

    async def __call__(
        self,
        server_id: str,
        tool_name: str,
        progress: float,
        message: str,
    ) -> None:
        """Emit progress event to SSE stream."""
        event = MCPProgressEvent.create(
            server_id=server_id,
            tool_name=tool_name,
            progress=progress,
            message=message,
            request_id=self._request_id,
        )
        await self._emitter.emit(event)


class MCPContentHandler:
    """
    Concrete content callback that emits events to EventEmitter.

    Streams multi-modal content (images, widgets, etc.) to the UI.
    """

    def __init__(self, emitter: EventEmitter, request_id: str | None = None):
        self._emitter = emitter
        self._request_id = request_id

    async def __call__(
        self,
        server_id: str,
        tool_name: str,
        content: Any,
        content_type: str,
    ) -> None:
        """Emit content event to SSE stream."""
        event = MCPContentEvent.create(
            server_id=server_id,
            tool_name=tool_name,
            content=content,
            content_type=content_type,
            request_id=self._request_id,
        )
        await self._emitter.emit(event)


class MCPErrorHandler:
    """
    Concrete error callback that emits events to EventEmitter.

    Reports MCP tool errors to the UI for display.
    """

    def __init__(self, emitter: EventEmitter, request_id: str | None = None):
        self._emitter = emitter
        self._request_id = request_id

    async def __call__(
        self,
        server_id: str,
        tool_name: str,
        error: str,
        error_type: str,
    ) -> None:
        """Emit error event to SSE stream."""
        event = MCPErrorEvent.create(
            server_id=server_id,
            tool_name=tool_name,
            error=error,
            error_type=error_type,
            request_id=self._request_id,
        )
        await self._emitter.emit(event)


class MCPElicitationHandler:
    """
    Concrete elicitation callback that manages user input requests.

    This handler:
    1. Registers the elicitation with the ElicitationManager
    2. Emits an event to the UI asking for input
    3. Waits for the user response (with timeout)
    4. Returns the response to the MCP tool

    The UI must call the elicitation response API endpoint to provide
    the user's answer, which will unblock this handler.
    """

    def __init__(
        self,
        emitter: EventEmitter,
        elicitation_manager: ElicitationManager,
        request_id: str | None = None,
    ):
        self._emitter = emitter
        self._manager = elicitation_manager
        self._request_id = request_id

    async def __call__(
        self,
        server_id: str,
        tool_name: str,
        request: ElicitationRequest,
    ) -> ElicitationResponse:
        """
        Handle elicitation request by emitting event and waiting for response.

        This method blocks until:
        1. User submits a response via API
        2. Timeout is reached
        3. Request is cancelled
        """
        # Create pending elicitation
        pending = await self._manager.create_elicitation(
            server_id=server_id,
            tool_name=tool_name,
            request=request,
        )

        # Emit event to UI
        event = MCPElicitationRequestEvent.create(
            server_id=server_id,
            tool_name=tool_name,
            elicitation_id=pending.elicitation_id,
            prompt=request.prompt,
            input_type=request.input_type,
            options=request.options,
            default=request.default,
            timeout_seconds=request.timeout_seconds,
            request_id=self._request_id,
        )
        await self._emitter.emit(event)

        # Wait for response with timeout
        try:
            response = await self._manager.wait_for_response(
                pending.elicitation_id,
                timeout=request.timeout_seconds,
            )
            return response
        except asyncio.TimeoutError:
            # Return cancelled response on timeout
            await self._manager.cancel_elicitation(pending.elicitation_id)
            return ElicitationResponse(
                request_id=pending.elicitation_id,
                value=None,
                cancelled=True,
            )


# =============================================================================
# CALLBACK CONTAINER
# =============================================================================


@dataclass
class MCPCallbacks:
    """Container for MCP callbacks - passed to session creation."""

    on_progress: MCPProgressCallback | None = None
    on_elicitation: MCPElicitationCallback | None = None
    on_content: MCPContentCallback | None = None
    on_error: MCPErrorCallback | None = None


def create_mcp_callbacks(
    emitter: EventEmitter,
    elicitation_manager: ElicitationManager,
    request_id: str | None = None,
) -> MCPCallbacks:
    """
    Factory function to create fully-wired MCP callbacks.

    Args:
        emitter: EventEmitter for SSE streaming
        elicitation_manager: Manager for user input requests
        request_id: Request ID for event correlation

    Returns:
        MCPCallbacks with all handlers configured
    """
    return MCPCallbacks(
        on_progress=MCPProgressHandler(emitter, request_id),
        on_content=MCPContentHandler(emitter, request_id),
        on_error=MCPErrorHandler(emitter, request_id),
        on_elicitation=MCPElicitationHandler(emitter, elicitation_manager, request_id),
    )
