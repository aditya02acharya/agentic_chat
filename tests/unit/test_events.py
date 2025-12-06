"""Tests for event system."""

import asyncio

import pytest

from agentic_chatbot.events.types import EventType
from agentic_chatbot.events.models import (
    Event,
    SupervisorThinkingEvent,
    ToolStartEvent,
    ResponseChunkEvent,
)
from agentic_chatbot.events.emitter import EventEmitter
from agentic_chatbot.events.bus import AsyncIOEventBus


class TestEventModels:
    """Tests for event models."""

    def test_event_to_sse(self):
        """Test SSE formatting."""
        event = Event(event_type=EventType.SUPERVISOR_THINKING, data={"message": "test"})
        sse = event.to_sse()
        assert sse.startswith("data: ")
        assert "supervisor.thinking" in sse
        assert "test" in sse

    def test_supervisor_thinking_event(self):
        """Test SupervisorThinkingEvent creation."""
        event = SupervisorThinkingEvent.create("Analyzing...", request_id="req-123")
        assert event.event_type == EventType.SUPERVISOR_THINKING
        assert event.data["message"] == "Analyzing..."
        assert event.request_id == "req-123"

    def test_tool_start_event(self):
        """Test ToolStartEvent creation."""
        event = ToolStartEvent.create("web_search", "Searching...")
        assert event.event_type == EventType.TOOL_START
        assert event.data["tool"] == "web_search"

    def test_response_chunk_event(self):
        """Test ResponseChunkEvent creation."""
        event = ResponseChunkEvent.create("Hello ")
        assert event.event_type == EventType.RESPONSE_CHUNK
        assert event.data["content"] == "Hello "


class TestEventEmitter:
    """Tests for event emitter."""

    @pytest.mark.asyncio
    async def test_emit_to_queue(self):
        """Test emitting events to queue."""
        queue: asyncio.Queue = asyncio.Queue()
        emitter = EventEmitter(queue)

        event = SupervisorThinkingEvent.create("test")
        await emitter.emit(event)

        assert not queue.empty()
        received = await queue.get()
        assert received.event_type == EventType.SUPERVISOR_THINKING

    @pytest.mark.asyncio
    async def test_subscribe_handler(self):
        """Test subscribing handlers."""
        emitter = EventEmitter()
        received_events = []

        async def handler(event: Event):
            received_events.append(event)

        emitter.subscribe(handler)

        event = ToolStartEvent.create("test_tool")
        await emitter.emit(event)

        assert len(received_events) == 1
        assert received_events[0].data["tool"] == "test_tool"


class TestEventBus:
    """Tests for event bus."""

    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        """Test basic pub/sub."""
        bus = AsyncIOEventBus()
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe("*", handler)

        event = SupervisorThinkingEvent.create("test")
        await bus.publish(event)

        # Wait for async handler
        await asyncio.sleep(0.1)

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_pattern_matching(self):
        """Test pattern-based subscription."""
        bus = AsyncIOEventBus()
        supervisor_events = []
        tool_events = []

        async def supervisor_handler(event: Event):
            supervisor_events.append(event)

        async def tool_handler(event: Event):
            tool_events.append(event)

        bus.subscribe("supervisor.*", supervisor_handler)
        bus.subscribe("tool.*", tool_handler)

        await bus.publish(SupervisorThinkingEvent.create("test"))
        await bus.publish(ToolStartEvent.create("test_tool"))

        await asyncio.sleep(0.1)

        assert len(supervisor_events) == 1
        assert len(tool_events) == 1

    def test_unsubscribe(self):
        """Test unsubscribing handlers."""
        bus = AsyncIOEventBus()

        def handler(event: Event):
            pass

        sub_id = bus.subscribe("*", handler)
        assert bus.unsubscribe(sub_id)
        assert not bus.unsubscribe(sub_id)  # Already removed
