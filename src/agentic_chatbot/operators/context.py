"""Operator context and result models."""

from typing import Any

from pydantic import BaseModel, Field


class OperatorContext(BaseModel):
    """Context provided to an operator for execution."""

    query: str = Field(description="The user's query or task description")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Additional parameters"
    )
    conversation_history: list[dict[str, str]] = Field(
        default_factory=list, description="Recent conversation messages"
    )
    previous_results: list[Any] = Field(
        default_factory=list, description="Results from previous operations"
    )
    tool_schemas: dict[str, dict] = Field(
        default_factory=dict, description="Schemas for available tools"
    )

    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a parameter value."""
        return self.params.get(key, default)


class OperatorResult(BaseModel):
    """Result from operator execution."""

    success: bool = True
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0

    @property
    def text(self) -> str:
        """Get text representation of output."""
        if isinstance(self.output, str):
            return self.output
        if self.output is not None:
            return str(self.output)
        return ""
