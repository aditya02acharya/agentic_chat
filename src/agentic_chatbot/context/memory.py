"""Conversation memory with window + summary."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in the conversation."""

    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationMemory:
    """
    Manages conversation history with window + summary.

    Window: Last N messages (configurable, default 5)
    Summary: Compressed summary of older messages

    This enables efficient context usage while maintaining
    conversation continuity.
    """

    def __init__(self, window_size: int = 5):
        """
        Initialize conversation memory.

        Args:
            window_size: Number of recent messages to keep in window
        """
        self.window_size = window_size
        self._messages: list[Message] = []
        self._summary: str = ""

    async def get(self, method: str) -> Any:
        """
        Get context based on method string.

        Supports:
        - "recent(N)": Last N messages
        - "summary": Summary of older messages
        - "all": Both summary and recent messages

        Args:
            method: Method string specifying what to retrieve

        Returns:
            Requested context data
        """
        if method.startswith("recent("):
            # Parse number from "recent(5)"
            try:
                n = int(method[7:-1])
            except ValueError:
                n = self.window_size
            return self.get_recent(n)
        elif method == "summary":
            return self._summary
        elif method == "all":
            return {
                "summary": self._summary,
                "recent": self.get_recent(self.window_size),
            }
        else:
            return None

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        """
        Add a message to the conversation.

        Args:
            role: Message role
            content: Message content
            **metadata: Additional metadata
        """
        self._messages.append(
            Message(role=role, content=content, metadata=metadata)
        )

    def get_recent(self, n: int | None = None) -> list[Message]:
        """
        Get the N most recent messages.

        Args:
            n: Number of messages (defaults to window_size)

        Returns:
            List of recent messages
        """
        n = n or self.window_size
        return self._messages[-n:]

    def get_recent_as_dicts(self, n: int | None = None) -> list[dict[str, Any]]:
        """Get recent messages as dictionaries."""
        return [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp.isoformat()}
            for m in self.get_recent(n)
        ]

    @property
    def summary(self) -> str:
        """Get conversation summary."""
        return self._summary

    def set_summary(self, summary: str) -> None:
        """Set conversation summary."""
        self._summary = summary

    async def compress_older_messages(self, llm_client: Any = None) -> None:
        """
        Compress older messages into summary.

        This should be called periodically to manage context size.

        Args:
            llm_client: Optional LLM client for summarization
        """
        if len(self._messages) <= self.window_size:
            return

        # Messages to compress
        to_compress = self._messages[:-self.window_size]

        if llm_client:
            # Use LLM to summarize
            messages_text = "\n".join(
                f"{m.role}: {m.content}" for m in to_compress
            )
            prompt = f"Summarize this conversation concisely:\n\n{messages_text}"

            response = await llm_client.complete(prompt, model="haiku")
            new_summary = response.content
        else:
            # Simple concatenation fallback
            new_summary = " ".join(m.content[:100] for m in to_compress[-3:])

        # Combine with existing summary
        if self._summary:
            self._summary = f"{self._summary}\n\n{new_summary}"
        else:
            self._summary = new_summary

        # Keep only window
        self._messages = self._messages[-self.window_size:]

    def clear(self) -> None:
        """Clear all messages and summary."""
        self._messages.clear()
        self._summary = ""

    def __len__(self) -> int:
        """Return total message count."""
        return len(self._messages)

    def format_for_prompt(self) -> str:
        """Format conversation for inclusion in prompts."""
        parts = []

        if self._summary:
            parts.append(f"Previous conversation summary:\n{self._summary}")

        if self._messages:
            recent_text = "\n".join(
                f"{m.role.title()}: {m.content}"
                for m in self.get_recent()
            )
            parts.append(f"Recent messages:\n{recent_text}")

        return "\n\n".join(parts) if parts else "No conversation history."
