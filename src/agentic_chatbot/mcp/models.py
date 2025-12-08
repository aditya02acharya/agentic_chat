"""MCP data models."""

from typing import Any

from pydantic import BaseModel, Field


class ServerInfo(BaseModel):
    """Information about an MCP server."""

    id: str
    name: str
    url: str
    description: str | None = None
    healthy: bool = True


class ToolSummary(BaseModel):
    """Summary info about a tool (for lazy loading)."""

    name: str
    description: str
    server_id: str


class ToolSchema(BaseModel):
    """Full tool schema with input parameters."""

    name: str
    description: str
    server_id: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ToolContent(BaseModel):
    """Content returned from a tool call."""

    content_type: str = "text/plain"
    data: Any = None
    is_error: bool = False


class ToolResult(BaseModel):
    """Complete result from a tool execution."""

    tool_name: str
    server_id: str
    success: bool = True
    content: list[ToolContent] = Field(default_factory=list)
    error: str | None = None
    duration_ms: float = 0.0

    @property
    def text(self) -> str:
        """Get text content from result."""
        texts = []
        for c in self.content:
            if isinstance(c.data, str):
                texts.append(c.data)
            elif c.data is not None:
                texts.append(str(c.data))
        return "\n".join(texts)
