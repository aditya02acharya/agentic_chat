"""Operator context and result models."""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from agentic_chatbot.mcp.models import ToolContent


class OperatorContext(BaseModel):
    """
    Context provided to an operator during execution.

    Built by ContextAssembler based on operator's context_requirements.
    """

    # Core query
    query: str = Field(..., description="The user's query")

    # Conversation context
    recent_messages: list[dict[str, Any]] = Field(
        default_factory=list, description="Recent conversation messages"
    )
    conversation_summary: str = Field("", description="Summary of older conversation")

    # Tool context (lazy loaded)
    tool_schemas: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Full schemas for required tools"
    )

    # Results from previous steps (for workflows)
    step_results: dict[str, Any] = Field(
        default_factory=dict, description="Results from previous workflow steps"
    )

    # Additional context
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Additional context data"
    )

    # Shared store reference (for advanced use)
    shared_store: dict[str, Any] = Field(
        default_factory=dict, description="Reference to shared store"
    )

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from extra context."""
        return self.extra.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in extra context."""
        self.extra[key] = value


class OperatorResult(BaseModel):
    """
    Result from operator execution.

    Supports both simple outputs and multi-modal content.
    """

    # Main output (text or structured data)
    output: Any = Field(None, description="Main output from the operator")

    # Multi-modal contents (images, widgets, etc.)
    contents: list[ToolContent] = Field(
        default_factory=list, description="Multi-modal content items"
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Execution metadata"
    )

    # Status
    success: bool = Field(True, description="Whether execution succeeded")
    error: str | None = Field(None, description="Error message if failed")

    # Token usage (for LLM operators)
    input_tokens: int = Field(0, description="Input tokens used")
    output_tokens: int = Field(0, description="Output tokens generated")

    @property
    def text_output(self) -> str:
        """Get output as text string."""
        if isinstance(self.output, str):
            return self.output
        if isinstance(self.output, dict):
            import json

            return json.dumps(self.output, indent=2)
        return str(self.output) if self.output else ""

    @property
    def has_contents(self) -> bool:
        """Check if result has multi-modal contents."""
        return len(self.contents) > 0

    @classmethod
    def success_result(
        cls,
        output: Any,
        contents: list[ToolContent] | None = None,
        **kwargs: Any,
    ) -> "OperatorResult":
        """Create successful result."""
        return cls(
            output=output,
            contents=contents or [],
            success=True,
            **kwargs,
        )

    @classmethod
    def error_result(cls, error: str, **kwargs: Any) -> "OperatorResult":
        """Create error result."""
        return cls(
            success=False,
            error=error,
            **kwargs,
        )
