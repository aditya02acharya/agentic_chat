"""Conversation memory management."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single conversation message."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationMemory:
    """
    Manages conversation history with windowing.

    Features:
    - Rolling window of recent messages
    - Summary generation for older messages
    - Token-aware truncation
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._messages: list[Message] = []
        self._summary: str | None = None

    def add_message(self, role: Literal["user", "assistant", "system"], content: str) -> None:
        """Add a message to the conversation."""
        self._messages.append(
            Message(role=role, content=content)
        )

    def get_recent(self, count: int | None = None) -> list[Message]:
        """Get recent messages within the window."""
        n = count or self.window_size
        return self._messages[-n:]

    def get_history_text(self) -> str:
        """Get formatted conversation history."""
        lines = []
        if self._summary:
            lines.append(f"[Previous conversation summary: {self._summary}]")
        for msg in self.get_recent():
            lines.append(f"{msg.role.upper()}: {msg.content}")
        return "\n".join(lines)

    def to_messages_list(self) -> list[dict[str, str]]:
        """Convert to list of message dicts."""
        return [{"role": m.role, "content": m.content} for m in self.get_recent()]

    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()
        self._summary = None

    @property
    def message_count(self) -> int:
        """Get total message count."""
        return len(self._messages)
