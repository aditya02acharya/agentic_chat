"""Agent Communication Protocol.

This module defines the event-driven communication protocol between agents,
operators, and tools. Agents don't work in isolation - they communicate
through well-defined events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentEventType(str, Enum):
    """Types of events agents can emit/consume."""

    # Lifecycle events
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    AGENT_BLOCKED = "agent.blocked"

    # Communication events
    MESSAGE_TO_USER = "message.to_user"
    MESSAGE_TO_AGENT = "message.to_agent"
    MESSAGE_BROADCAST = "message.broadcast"

    # Clarification events
    CLARIFICATION_NEEDED = "clarification.needed"
    CLARIFICATION_RECEIVED = "clarification.received"

    # Delegation events
    TASK_DELEGATED = "task.delegated"
    TASK_ACCEPTED = "task.accepted"
    TASK_REJECTED = "task.rejected"
    TASK_PROGRESS = "task.progress"
    TASK_RESULT = "task.result"

    # Tool events
    TOOL_SELECTED = "tool.selected"
    TOOL_INVOKED = "tool.invoked"
    TOOL_RESULT = "tool.result"
    TOOL_ERROR = "tool.error"

    # Context events
    CONTEXT_UPDATED = "context.updated"
    CONTEXT_REQUESTED = "context.requested"
    CONTEXT_SHARED = "context.shared"

    # Control events
    THINKING_BUDGET_SET = "control.thinking_budget"
    PRIORITY_CHANGED = "control.priority"
    TIMEOUT_WARNING = "control.timeout_warning"
    CANCEL_REQUESTED = "control.cancel"


class AgentEvent(BaseModel):
    """
    Base event for agent communication.

    All agent communication happens through events, enabling:
    - Loose coupling between agents
    - Event sourcing for replay/debugging
    - Flexible routing and filtering
    """

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: AgentEventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Source and target
    source_agent: str = Field(description="Agent that emitted this event")
    target_agent: str | None = Field(default=None, description="Specific target, or None for broadcast")

    # Payload
    payload: dict[str, Any] = Field(default_factory=dict)

    # Metadata
    conversation_id: str = Field(default="")
    request_id: str = Field(default="")
    correlation_id: str | None = Field(default=None, description="Links related events")
    parent_event_id: str | None = Field(default=None, description="For event chains")

    # Priority and control
    priority: int = Field(default=3, ge=1, le=5, description="1=highest")
    requires_ack: bool = Field(default=False, description="Requires acknowledgment")


class DelegationContext(BaseModel):
    """
    Context passed when supervisor delegates to an agent/operator.

    The supervisor has fine-grained control over how delegates operate.
    """

    # Task definition
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    task_description: str = Field(description="What the delegate should do")
    task_goal: str = Field(description="Expected outcome")
    task_scope: str = Field(default="", description="What's in/out of scope")

    # Query understanding context
    query_understanding: dict[str, Any] | None = Field(default=None, description="Parsed query understanding")

    # Resource controls
    enable_thinking: bool = Field(default=True, description="Allow extended thinking")
    thinking_budget: int = Field(default=10000, description="Max thinking tokens")
    max_tokens: int = Field(default=4096, description="Max output tokens")
    timeout_seconds: float = Field(default=60.0, description="Task timeout")

    # Tool access controls
    allowed_tools: list[str] | None = Field(default=None, description="Whitelist of tools, None=all")
    denied_tools: list[str] = Field(default_factory=list, description="Blacklist of tools")
    can_delegate_further: bool = Field(default=False, description="Can this agent delegate to others")

    # Communication controls
    can_clarify: bool = Field(default=True, description="Can request clarification from user")
    can_emit_progress: bool = Field(default=True, description="Can emit progress events")
    stream_output: bool = Field(default=True, description="Stream output to user")

    # Context sharing
    shared_context: dict[str, Any] = Field(default_factory=dict, description="Context shared with delegate")
    context_filter: list[str] = Field(default_factory=list, description="What context to include")

    # Priority
    priority: int = Field(default=3, ge=1, le=5, description="Task priority, 1=highest")


class AgentCapabilities(BaseModel):
    """
    Declares what an agent can do.

    Agents advertise their capabilities so the supervisor
    can make informed delegation decisions.
    """

    agent_id: str
    agent_name: str
    description: str

    # What it can do
    supported_intents: list[str] = Field(default_factory=list)
    supported_domains: list[str] = Field(default_factory=list)
    supported_tasks: list[str] = Field(default_factory=list)

    # Resource requirements
    requires_tools: bool = False
    requires_context: bool = False
    requires_thinking: bool = False

    # Communication
    can_clarify: bool = False
    can_delegate: bool = False
    can_stream: bool = False

    # Limits
    max_input_tokens: int = 100000
    max_output_tokens: int = 8192
    typical_latency_ms: int = 5000


class AgentResponse(BaseModel):
    """
    Response from an agent after task execution.
    """

    task_id: str
    agent_id: str
    status: str = Field(description="success, partial, failed, blocked, clarification_needed")

    # Results
    result: Any = None
    result_summary: str = Field(default="", description="Brief summary for context efficiency")

    # If clarification needed
    clarification_request: dict[str, Any] | None = None

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0

    # Metadata
    execution_time_ms: int = 0
    events_emitted: list[str] = Field(default_factory=list)


# Type for event handlers
EventHandler = Callable[[AgentEvent], Awaitable[None]]


class AgentEventBus:
    """
    Event bus for agent communication.

    Agents subscribe to event types and receive events asynchronously.
    This enables loose coupling and flexible communication patterns.
    """

    def __init__(self):
        self._handlers: dict[AgentEventType, list[EventHandler]] = {}
        self._agent_handlers: dict[str, list[EventHandler]] = {}
        self._event_history: list[AgentEvent] = []
        self._max_history = 1000

    def subscribe(
        self,
        event_type: AgentEventType | None = None,
        agent_id: str | None = None,
        handler: EventHandler = None,
    ) -> None:
        """
        Subscribe to events.

        Args:
            event_type: Subscribe to specific event type
            agent_id: Subscribe to events for specific agent
            handler: Async handler function
        """
        if event_type:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

        if agent_id:
            if agent_id not in self._agent_handlers:
                self._agent_handlers[agent_id] = []
            self._agent_handlers[agent_id].append(handler)

    def unsubscribe(
        self,
        event_type: AgentEventType | None = None,
        agent_id: str | None = None,
        handler: EventHandler = None,
    ) -> None:
        """Unsubscribe from events."""
        if event_type and event_type in self._handlers:
            self._handlers[event_type] = [h for h in self._handlers[event_type] if h != handler]

        if agent_id and agent_id in self._agent_handlers:
            self._agent_handlers[agent_id] = [h for h in self._agent_handlers[agent_id] if h != handler]

    async def publish(self, event: AgentEvent) -> None:
        """
        Publish an event to all subscribers.

        Events are delivered to:
        1. All handlers subscribed to the event type
        2. All handlers subscribed to the target agent
        """
        # Record in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        # Notify event type subscribers
        if event.event_type in self._handlers:
            for handler in self._handlers[event.event_type]:
                try:
                    await handler(event)
                except Exception:
                    pass  # Don't let one handler break others

        # Notify agent-specific subscribers
        if event.target_agent and event.target_agent in self._agent_handlers:
            for handler in self._agent_handlers[event.target_agent]:
                try:
                    await handler(event)
                except Exception:
                    pass

    def get_history(
        self,
        event_type: AgentEventType | None = None,
        agent_id: str | None = None,
        limit: int = 100,
    ) -> list[AgentEvent]:
        """Get event history with optional filtering."""
        events = self._event_history

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if agent_id:
            events = [e for e in events if e.source_agent == agent_id or e.target_agent == agent_id]

        return events[-limit:]


# Global event bus instance
_event_bus: AgentEventBus | None = None


def get_event_bus() -> AgentEventBus:
    """Get or create the global event bus."""
    global _event_bus
    if _event_bus is None:
        _event_bus = AgentEventBus()
    return _event_bus
