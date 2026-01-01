"""Base class for local tools.

Local tools execute in-process without network calls, providing:
- Zero network latency
- Self-awareness capabilities (version, capabilities, release notes)
- Simple utilities (date/time, calculations)
- Introspection (what tools/operators are available)

Design Philosophy:
- Local tools are DATA FETCHERS, not execution strategies (that's operators)
- They follow the same ToolSummary/ToolResult interface as MCP tools
- Supervisor sees them identically to remote tools (unified interface)
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from agentic_chatbot.mcp.models import (
    MessagingCapabilities,
    OutputDataType,
    ToolSummary,
    ToolSchema,
    ToolResult,
    ToolResultStatus,
    ToolContent,
)


class LocalToolContext(BaseModel):
    """Context provided to local tools during execution."""

    # Parameters passed to the tool
    params: dict[str, Any] = Field(default_factory=dict)

    # Request context
    request_id: str | None = None
    conversation_id: str | None = None

    # Optional: access to registries for introspection tools
    operator_registry: Any | None = None
    mcp_registry: Any | None = None
    local_tool_registry: Any | None = None

    # Document service for document loading tools
    document_service: Any | None = None


class LocalTool(ABC):
    """
    Base class for local tools.

    Local tools are simple, fast, in-process tools that don't require
    network calls. They're ideal for:
    - Self-awareness (capabilities, version info)
    - Introspection (listing available tools/operators)
    - Simple utilities

    Unlike operators (which are execution strategies), local tools
    are pure data fetchers that return ToolResult.

    Example:
        @LocalToolRegistry.register
        class DateTimeTool(LocalTool):
            name = "datetime"
            description = "Get current date and time"

            async def execute(self, context: LocalToolContext) -> ToolResult:
                from datetime import datetime
                now = datetime.now().isoformat()
                return ToolResult.success(self.name, [ToolContent.text(now)])
    """

    # Metadata - must be set by subclasses
    name: str
    description: str

    # Input schema for the tool (JSON Schema format)
    input_schema: dict[str, Any] = {}

    # Messaging capabilities (what this tool can return/do)
    messaging: MessagingCapabilities = MessagingCapabilities.default()

    # Whether this tool requires access to registries
    needs_introspection: bool = False

    @abstractmethod
    async def execute(self, context: LocalToolContext) -> ToolResult:
        """
        Execute the tool and return a result.

        Args:
            context: Execution context with parameters and optional registries

        Returns:
            ToolResult with the tool's output
        """
        pass

    def get_summary(self) -> ToolSummary:
        """Get tool summary for supervisor context."""
        return ToolSummary(
            name=f"local:{self.name}",
            description=self.description,
            server_id="local",
            messaging=self.messaging,
        )

    def get_schema(self) -> ToolSchema:
        """Get full tool schema."""
        return ToolSchema(
            name=f"local:{self.name}",
            description=self.description,
            server_id="local",
            input_schema=self.input_schema,
            messaging=self.messaging,
        )

    @classmethod
    def success(
        cls,
        tool_name: str,
        content: str | dict | list,
        content_type: str = "text/plain",
    ) -> ToolResult:
        """Helper to create a successful result."""
        if isinstance(content, str):
            contents = [ToolContent(content_type=content_type, data=content)]
        elif isinstance(content, dict):
            import json
            contents = [ToolContent(content_type="application/json", data=json.dumps(content))]
        elif isinstance(content, list):
            contents = content  # Assume list of ToolContent
        else:
            contents = [ToolContent(content_type="text/plain", data=str(content))]

        return ToolResult(
            tool_name=tool_name,
            status=ToolResultStatus.SUCCESS,
            contents=contents,
            duration_ms=0.0,
        )

    @classmethod
    def error(cls, tool_name: str, error_message: str) -> ToolResult:
        """Helper to create an error result."""
        return ToolResult(
            tool_name=tool_name,
            status=ToolResultStatus.ERROR,
            contents=[],
            error=error_message,
            duration_ms=0.0,
        )
