"""Event data models."""

import json
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .types import EventType


class Event(BaseModel):
    """Base event model."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize event to JSON string."""
        return json.dumps({
            "id": self.id,
            "type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
            "data": self.data,
        })

    def to_sse(self) -> str:
        """Format event for SSE stream."""
        return f"event: {self.event_type.value}\ndata: {self.to_json()}\n\n"


class ProgressEvent(Event):
    """Event for progress updates."""

    event_type: EventType = EventType.TOOL_PROGRESS
    progress: float = 0.0
    message: str = ""

    def model_post_init(self, __context: Any) -> None:
        self.data["progress"] = self.progress
        self.data["message"] = self.message


class ContentEvent(Event):
    """Event for content streaming."""

    event_type: EventType = EventType.RESPONSE_CHUNK
    content: str = ""
    content_type: str = "text/plain"

    def model_post_init(self, __context: Any) -> None:
        self.data["content"] = self.content
        self.data["content_type"] = self.content_type


class ErrorEvent(Event):
    """Event for error reporting."""

    event_type: EventType = EventType.ERROR
    error_code: str = "unknown"
    error_message: str = ""
    recoverable: bool = True

    def model_post_init(self, __context: Any) -> None:
        self.data["error_code"] = self.error_code
        self.data["error_message"] = self.error_message
        self.data["recoverable"] = self.recoverable
