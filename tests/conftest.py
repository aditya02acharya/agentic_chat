"""Pytest fixtures for testing."""

import asyncio
from typing import AsyncIterator, Any

import pytest
import pytest_asyncio

from agentic_chatbot.config.settings import Settings
from agentic_chatbot.context.memory import ConversationMemory
from agentic_chatbot.context.results import ResultStore
from agentic_chatbot.context.actions import ActionHistory
from agentic_chatbot.events.emitter import EventEmitter
from agentic_chatbot.events.models import Event
from agentic_chatbot.mcp.mock.server import MockMCPServer
from agentic_chatbot.operators.context import OperatorContext


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings(
        anthropic_api_key="test-key",
        mcp_discovery_url="http://localhost:8080/servers",
        log_level="DEBUG",
    )


@pytest_asyncio.fixture
async def event_queue() -> asyncio.Queue[Event]:
    """Create event queue for testing."""
    return asyncio.Queue()


@pytest_asyncio.fixture
async def event_emitter(event_queue: asyncio.Queue[Event]) -> EventEmitter:
    """Create event emitter for testing."""
    return EventEmitter(event_queue)


@pytest.fixture
def mock_mcp_server() -> MockMCPServer:
    """Create mock MCP server for testing."""
    return MockMCPServer(server_id="test_server")


@pytest.fixture
def conversation_memory() -> ConversationMemory:
    """Create conversation memory for testing."""
    return ConversationMemory(window_size=5)


@pytest.fixture
def result_store() -> ResultStore:
    """Create result store for testing."""
    return ResultStore()


@pytest.fixture
def action_history() -> ActionHistory:
    """Create action history for testing."""
    return ActionHistory()


@pytest.fixture
def operator_context() -> OperatorContext:
    """Create basic operator context for testing."""
    return OperatorContext(query="test query")


@pytest.fixture
def shared_store(
    event_queue: asyncio.Queue[Event],
    conversation_memory: ConversationMemory,
    result_store: ResultStore,
    action_history: ActionHistory,
) -> dict[str, Any]:
    """Create shared store for flow testing."""
    return {
        "user_query": "Test query",
        "conversation_id": "test-conv-123",
        "request_id": "test-req-456",
        "event_queue": event_queue,
        "memory": conversation_memory,
        "result_store": result_store,
        "action_history": action_history,
        "results": {
            "tool_outputs": [],
            "workflow_output": None,
            "synthesis": None,
            "final_response": None,
        },
        "supervisor": {
            "state": None,
            "current_decision": None,
        },
        "mcp": {
            "server_registry": None,
            "session_manager": None,
        },
    }
