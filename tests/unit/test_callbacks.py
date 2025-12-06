"""Tests for MCP callback implementations."""

import asyncio
import pytest

from agentic_chatbot.events.emitter import EventEmitter
from agentic_chatbot.events.types import EventType
from agentic_chatbot.mcp.callbacks import (
    ElicitationManager,
    MCPProgressHandler,
    MCPContentHandler,
    MCPErrorHandler,
    MCPElicitationHandler,
    create_mcp_callbacks,
)
from agentic_chatbot.mcp.models import ElicitationRequest, ElicitationResponse


class TestElicitationManager:
    """Tests for ElicitationManager."""

    @pytest.fixture
    def manager(self) -> ElicitationManager:
        return ElicitationManager()

    @pytest.fixture
    def sample_request(self) -> ElicitationRequest:
        return ElicitationRequest(
            request_id="test-123",
            prompt="Please confirm this action",
            input_type="confirm",
            timeout_seconds=30.0,
        )

    @pytest.mark.asyncio
    async def test_create_elicitation(
        self, manager: ElicitationManager, sample_request: ElicitationRequest
    ) -> None:
        """Test creating a pending elicitation."""
        pending = await manager.create_elicitation(
            server_id="test-server",
            tool_name="test-tool",
            request=sample_request,
        )

        assert pending.elicitation_id == "test-123"
        assert pending.server_id == "test-server"
        assert pending.tool_name == "test-tool"
        assert pending.request == sample_request
        assert not pending.future.done()

    @pytest.mark.asyncio
    async def test_submit_response(
        self, manager: ElicitationManager, sample_request: ElicitationRequest
    ) -> None:
        """Test submitting a response to an elicitation."""
        pending = await manager.create_elicitation(
            server_id="test-server",
            tool_name="test-tool",
            request=sample_request,
        )

        success = await manager.submit_response("test-123", value=True)
        assert success

        # Future should now be resolved
        assert pending.future.done()
        response = pending.future.result()
        assert response.value is True
        assert not response.cancelled

    @pytest.mark.asyncio
    async def test_submit_response_not_found(
        self, manager: ElicitationManager
    ) -> None:
        """Test submitting response to non-existent elicitation."""
        success = await manager.submit_response("nonexistent", value="test")
        assert not success

    @pytest.mark.asyncio
    async def test_cancel_elicitation(
        self, manager: ElicitationManager, sample_request: ElicitationRequest
    ) -> None:
        """Test cancelling an elicitation."""
        pending = await manager.create_elicitation(
            server_id="test-server",
            tool_name="test-tool",
            request=sample_request,
        )

        success = await manager.cancel_elicitation("test-123")
        assert success

        response = pending.future.result()
        assert response.cancelled
        assert response.value is None

    @pytest.mark.asyncio
    async def test_wait_for_response(
        self, manager: ElicitationManager, sample_request: ElicitationRequest
    ) -> None:
        """Test waiting for a response."""
        await manager.create_elicitation(
            server_id="test-server",
            tool_name="test-tool",
            request=sample_request,
        )

        # Submit response in background
        async def submit_later() -> None:
            await asyncio.sleep(0.1)
            await manager.submit_response("test-123", value="user input")

        asyncio.create_task(submit_later())

        response = await manager.wait_for_response("test-123", timeout=5.0)
        assert response.value == "user input"
        assert not response.cancelled

    @pytest.mark.asyncio
    async def test_wait_for_response_timeout(
        self, manager: ElicitationManager, sample_request: ElicitationRequest
    ) -> None:
        """Test timeout when waiting for response."""
        await manager.create_elicitation(
            server_id="test-server",
            tool_name="test-tool",
            request=sample_request,
        )

        with pytest.raises(asyncio.TimeoutError):
            await manager.wait_for_response("test-123", timeout=0.1)

    @pytest.mark.asyncio
    async def test_get_all_pending(
        self, manager: ElicitationManager
    ) -> None:
        """Test getting all pending elicitations."""
        request1 = ElicitationRequest(
            request_id="req-1",
            prompt="First question",
        )
        request2 = ElicitationRequest(
            request_id="req-2",
            prompt="Second question",
        )

        await manager.create_elicitation("server-1", "tool-1", request1)
        await manager.create_elicitation("server-2", "tool-2", request2)

        pending = manager.get_all_pending()
        assert len(pending) == 2
        ids = {p.elicitation_id for p in pending}
        assert ids == {"req-1", "req-2"}

    @pytest.mark.asyncio
    async def test_cleanup_expired(
        self, manager: ElicitationManager
    ) -> None:
        """Test cleaning up expired elicitations."""
        request = ElicitationRequest(
            request_id="old-request",
            prompt="Old question",
        )

        pending = await manager.create_elicitation("server", "tool", request)
        # Simulate old timestamp by modifying created_at
        pending.created_at = asyncio.get_event_loop().time() - 400

        cleaned = await manager.cleanup_expired(max_age_seconds=300)
        assert cleaned == 1
        assert len(manager.get_all_pending()) == 0


class TestMCPHandlers:
    """Tests for MCP callback handlers."""

    @pytest.fixture
    def event_queue(self) -> asyncio.Queue:
        return asyncio.Queue()

    @pytest.fixture
    def emitter(self, event_queue: asyncio.Queue) -> EventEmitter:
        return EventEmitter(event_queue)

    @pytest.mark.asyncio
    async def test_progress_handler(
        self, emitter: EventEmitter, event_queue: asyncio.Queue
    ) -> None:
        """Test MCPProgressHandler emits correct events."""
        handler = MCPProgressHandler(emitter, request_id="req-123")

        await handler(
            server_id="server-1",
            tool_name="search",
            progress=0.5,
            message="Halfway done",
        )

        event = await event_queue.get()
        assert event.event_type == EventType.MCP_PROGRESS
        assert event.data["server_id"] == "server-1"
        assert event.data["tool"] == "search"
        assert event.data["progress"] == 0.5
        assert event.data["message"] == "Halfway done"
        assert event.request_id == "req-123"

    @pytest.mark.asyncio
    async def test_content_handler(
        self, emitter: EventEmitter, event_queue: asyncio.Queue
    ) -> None:
        """Test MCPContentHandler emits correct events."""
        handler = MCPContentHandler(emitter, request_id="req-456")

        await handler(
            server_id="server-2",
            tool_name="image_gen",
            content="base64data...",
            content_type="image/png",
        )

        event = await event_queue.get()
        assert event.event_type == EventType.MCP_CONTENT
        assert event.data["server_id"] == "server-2"
        assert event.data["tool"] == "image_gen"
        assert event.data["content"] == "base64data..."
        assert event.data["content_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_error_handler(
        self, emitter: EventEmitter, event_queue: asyncio.Queue
    ) -> None:
        """Test MCPErrorHandler emits correct events."""
        handler = MCPErrorHandler(emitter, request_id="req-789")

        await handler(
            server_id="server-3",
            tool_name="web_search",
            error="Connection refused",
            error_type="connection",
        )

        event = await event_queue.get()
        assert event.event_type == EventType.MCP_ERROR
        assert event.data["server_id"] == "server-3"
        assert event.data["tool"] == "web_search"
        assert event.data["error"] == "Connection refused"
        assert event.data["error_type"] == "connection"

    @pytest.mark.asyncio
    async def test_elicitation_handler(
        self, emitter: EventEmitter, event_queue: asyncio.Queue
    ) -> None:
        """Test MCPElicitationHandler emits event and waits for response."""
        manager = ElicitationManager()
        handler = MCPElicitationHandler(emitter, manager, request_id="req-elicit")

        request = ElicitationRequest(
            request_id="elicit-1",
            prompt="Do you want to continue?",
            input_type="confirm",
            timeout_seconds=5.0,
        )

        # Submit response in background
        async def submit_later() -> None:
            await asyncio.sleep(0.1)
            await manager.submit_response("elicit-1", value=True)

        asyncio.create_task(submit_later())

        response = await handler(
            server_id="server-4",
            tool_name="confirm_tool",
            request=request,
        )

        # Check event was emitted
        event = await event_queue.get()
        assert event.event_type == EventType.MCP_ELICITATION
        assert event.data["elicitation_id"] == "elicit-1"
        assert event.data["prompt"] == "Do you want to continue?"
        assert event.data["input_type"] == "confirm"

        # Check response
        assert response.value is True
        assert not response.cancelled

    @pytest.mark.asyncio
    async def test_elicitation_handler_timeout(
        self, emitter: EventEmitter, event_queue: asyncio.Queue
    ) -> None:
        """Test MCPElicitationHandler handles timeout."""
        manager = ElicitationManager()
        handler = MCPElicitationHandler(emitter, manager, request_id="req-timeout")

        request = ElicitationRequest(
            request_id="elicit-timeout",
            prompt="Quick question",
            timeout_seconds=0.1,  # Very short timeout
        )

        response = await handler(
            server_id="server-5",
            tool_name="quick_tool",
            request=request,
        )

        assert response.cancelled
        assert response.value is None


class TestCreateMCPCallbacks:
    """Tests for create_mcp_callbacks factory function."""

    @pytest.mark.asyncio
    async def test_creates_all_callbacks(self) -> None:
        """Test that factory creates all callback handlers."""
        queue: asyncio.Queue = asyncio.Queue()
        emitter = EventEmitter(queue)
        manager = ElicitationManager()

        callbacks = create_mcp_callbacks(emitter, manager, request_id="test")

        assert callbacks.on_progress is not None
        assert callbacks.on_content is not None
        assert callbacks.on_error is not None
        assert callbacks.on_elicitation is not None

    @pytest.mark.asyncio
    async def test_callbacks_emit_events(self) -> None:
        """Test that created callbacks properly emit events."""
        queue: asyncio.Queue = asyncio.Queue()
        emitter = EventEmitter(queue)
        manager = ElicitationManager()

        callbacks = create_mcp_callbacks(emitter, manager, request_id="factory-test")

        # Test progress callback
        await callbacks.on_progress(
            server_id="s1",
            tool_name="t1",
            progress=0.25,
            message="Quarter done",
        )

        event = await queue.get()
        assert event.event_type == EventType.MCP_PROGRESS
        assert event.request_id == "factory-test"
