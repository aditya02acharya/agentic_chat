"""Event data models."""

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agentic_chatbot.events.types import EventType


class Event(BaseModel):
    """Base event model for SSE streaming."""

    event_type: EventType = Field(..., alias="type")
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str | None = None

    model_config = {"populate_by_name": True}

    def to_sse(self) -> str:
        """Format event for SSE streaming."""
        payload = {"type": self.event_type.value, **self.data}
        return f"data: {json.dumps(payload)}\n\n"

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({"type": self.event_type.value, **self.data})


# =============================================================================
# SUPERVISOR EVENTS
# =============================================================================


class SupervisorThinkingEvent(Event):
    """Supervisor is analyzing the query."""

    event_type: EventType = EventType.SUPERVISOR_THINKING

    @classmethod
    def create(cls, message: str, request_id: str | None = None) -> "SupervisorThinkingEvent":
        return cls(data={"message": message}, request_id=request_id)


class SupervisorDecidedEvent(Event):
    """Supervisor has made a decision."""

    event_type: EventType = EventType.SUPERVISOR_DECIDED

    @classmethod
    def create(
        cls, action: str, message: str, request_id: str | None = None
    ) -> "SupervisorDecidedEvent":
        return cls(data={"action": action, "message": message}, request_id=request_id)


# =============================================================================
# TOOL EVENTS
# =============================================================================


class ToolStartEvent(Event):
    """Tool execution started."""

    event_type: EventType = EventType.TOOL_START

    @classmethod
    def create(
        cls, tool: str, message: str = "", request_id: str | None = None
    ) -> "ToolStartEvent":
        return cls(data={"tool": tool, "message": message}, request_id=request_id)


class ToolProgressEvent(Event):
    """Tool execution progress update."""

    event_type: EventType = EventType.TOOL_PROGRESS

    @classmethod
    def create(
        cls, tool: str, progress: float, message: str = "", request_id: str | None = None
    ) -> "ToolProgressEvent":
        return cls(
            data={"tool": tool, "progress": progress, "message": message},
            request_id=request_id,
        )


class ToolContentEvent(Event):
    """Tool returned content (text, images, widgets)."""

    event_type: EventType = EventType.TOOL_CONTENT

    @classmethod
    def create(
        cls,
        tool: str,
        content_type: str,
        data: Any,
        encoding: str | None = None,
        metadata: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> "ToolContentEvent":
        event_data = {
            "tool": tool,
            "content_type": content_type,
            "data": data,
        }
        if encoding:
            event_data["encoding"] = encoding
        if metadata:
            event_data["metadata"] = metadata
        return cls(data=event_data, request_id=request_id)


class ToolCompleteEvent(Event):
    """Tool execution completed."""

    event_type: EventType = EventType.TOOL_COMPLETE

    @classmethod
    def create(
        cls, tool: str, content_count: int = 1, request_id: str | None = None
    ) -> "ToolCompleteEvent":
        return cls(data={"tool": tool, "content_count": content_count}, request_id=request_id)


class ToolErrorEvent(Event):
    """Tool execution failed."""

    event_type: EventType = EventType.TOOL_ERROR

    @classmethod
    def create(
        cls, tool: str, error: str, error_type: str = "execution", request_id: str | None = None
    ) -> "ToolErrorEvent":
        return cls(
            data={"tool": tool, "error": error, "error_type": error_type},
            request_id=request_id,
        )


# =============================================================================
# WORKFLOW EVENTS
# =============================================================================


class WorkflowCreatedEvent(Event):
    """Workflow was created."""

    event_type: EventType = EventType.WORKFLOW_CREATED

    @classmethod
    def create(
        cls, goal: str, steps: int, request_id: str | None = None
    ) -> "WorkflowCreatedEvent":
        return cls(data={"goal": goal, "steps": steps}, request_id=request_id)


class WorkflowStepStartEvent(Event):
    """Workflow step started."""

    event_type: EventType = EventType.WORKFLOW_STEP_START

    @classmethod
    def create(
        cls, step: int, name: str, request_id: str | None = None
    ) -> "WorkflowStepStartEvent":
        return cls(data={"step": step, "name": name}, request_id=request_id)


class WorkflowStepCompleteEvent(Event):
    """Workflow step completed."""

    event_type: EventType = EventType.WORKFLOW_STEP_COMPLETE

    @classmethod
    def create(cls, step: int, request_id: str | None = None) -> "WorkflowStepCompleteEvent":
        return cls(data={"step": step}, request_id=request_id)


class WorkflowCompleteEvent(Event):
    """Workflow completed."""

    event_type: EventType = EventType.WORKFLOW_COMPLETE

    @classmethod
    def create(cls, request_id: str | None = None) -> "WorkflowCompleteEvent":
        return cls(data={}, request_id=request_id)


# =============================================================================
# RESPONSE EVENTS
# =============================================================================


class ResponseChunkEvent(Event):
    """Response chunk for streaming."""

    event_type: EventType = EventType.RESPONSE_CHUNK

    @classmethod
    def create(cls, content: str, request_id: str | None = None) -> "ResponseChunkEvent":
        return cls(data={"content": content}, request_id=request_id)


class ResponseDoneEvent(Event):
    """Response streaming complete."""

    event_type: EventType = EventType.RESPONSE_DONE

    @classmethod
    def create(cls, request_id: str | None = None) -> "ResponseDoneEvent":
        return cls(data={}, request_id=request_id)


# =============================================================================
# USER INTERACTION EVENTS
# =============================================================================


class ClarifyRequestEvent(Event):
    """Request clarification from user."""

    event_type: EventType = EventType.CLARIFY_REQUEST

    @classmethod
    def create(cls, question: str, request_id: str | None = None) -> "ClarifyRequestEvent":
        return cls(data={"question": question}, request_id=request_id)


class ErrorEvent(Event):
    """General error event."""

    event_type: EventType = EventType.ERROR

    @classmethod
    def create(
        cls, error: str, error_type: str = "general", request_id: str | None = None
    ) -> "ErrorEvent":
        return cls(data={"error": error, "error_type": error_type}, request_id=request_id)
