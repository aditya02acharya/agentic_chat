"""MCP data models for tool communication."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# MESSAGING CAPABILITIES
# =============================================================================


class OutputDataType(str, Enum):
    """Data types that tools/operators can return."""

    TEXT = "text"  # Plain text or markdown
    HTML = "html"  # HTML content
    IMAGE = "image"  # Image data (base64 encoded)
    WIDGET = "widget"  # Interactive widget/component
    JSON = "json"  # Structured JSON data
    BINARY = "binary"  # Binary data
    MIXED = "mixed"  # Multiple data types


class MessagingCapabilities(BaseModel):
    """
    Describes the messaging capabilities of a tool or operator.

    This metadata tells the system what a tool/operator can do in terms of
    communication and response handling.
    """

    # Output data types this tool/operator can return
    output_types: list[OutputDataType] = Field(
        default_factory=lambda: [OutputDataType.TEXT],
        description="Data types that can be returned (text, image, widget, etc.)",
    )

    # Progress reporting capability
    supports_progress: bool = Field(
        False,
        description="Whether the tool/operator can report intermediate progress updates",
    )

    # Elicitation (user input) capability
    supports_elicitation: bool = Field(
        False,
        description="Whether the tool/operator can request user input during execution",
    )

    # Direct response capability (bypass writer)
    supports_direct_response: bool = Field(
        False,
        description="Whether the tool/operator can send responses directly to user, bypassing the writer",
    )

    # Streaming content capability
    supports_streaming: bool = Field(
        False,
        description="Whether the tool/operator can stream content incrementally",
    )

    @classmethod
    def default(cls) -> "MessagingCapabilities":
        """Create default capabilities (text only, no special features)."""
        return cls()

    @classmethod
    def full(cls) -> "MessagingCapabilities":
        """Create full capabilities (all features enabled)."""
        return cls(
            output_types=[
                OutputDataType.TEXT,
                OutputDataType.HTML,
                OutputDataType.IMAGE,
                OutputDataType.WIDGET,
                OutputDataType.JSON,
            ],
            supports_progress=True,
            supports_elicitation=True,
            supports_direct_response=True,
            supports_streaming=True,
        )

    @classmethod
    def widget_capable(cls) -> "MessagingCapabilities":
        """Create capabilities for widget-returning tools."""
        return cls(
            output_types=[OutputDataType.TEXT, OutputDataType.WIDGET, OutputDataType.HTML],
            supports_direct_response=True,
        )

    @classmethod
    def image_capable(cls) -> "MessagingCapabilities":
        """Create capabilities for image-returning tools."""
        return cls(
            output_types=[OutputDataType.TEXT, OutputDataType.IMAGE],
            supports_progress=True,
        )


# =============================================================================
# SERVER AND TOOL METADATA
# =============================================================================


class MCPServerInfo(BaseModel):
    """Information about an MCP server."""

    id: str = Field(..., description="Unique server identifier")
    name: str = Field(..., description="Human-readable server name")
    url: str = Field(..., description="Server base URL")
    description: str = Field("", description="Server description")
    version: str = Field("1.0.0", description="Server version")
    tools: list[str] = Field(default_factory=list, description="Tool names hosted by this server")
    healthy: bool = Field(True, description="Whether server is healthy")


class ToolSummary(BaseModel):
    """
    Summary information about a tool (for supervisor context).

    This is the lightweight version loaded at startup.
    Includes messaging capabilities metadata for routing decisions.
    """

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Short description of the tool")
    server_id: str = Field(..., description="ID of server hosting this tool")

    # Messaging capabilities metadata
    messaging: MessagingCapabilities = Field(
        default_factory=MessagingCapabilities.default,
        description="Messaging capabilities of this tool",
    )


class ToolSchema(BaseModel):
    """
    Full schema for a tool (lazy loaded on demand).

    Contains the complete JSON schema for tool inputs and messaging capabilities.
    """

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    server_id: str = Field(..., description="ID of server hosting this tool")
    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for tool inputs"
    )

    # Messaging capabilities metadata
    messaging: MessagingCapabilities = Field(
        default_factory=MessagingCapabilities.default,
        description="Messaging capabilities of this tool",
    )


# =============================================================================
# CONTENT TYPES
# =============================================================================


class ContentType(str, Enum):
    """Supported content types for tool outputs."""

    TEXT = "text/plain"
    MARKDOWN = "text/markdown"
    HTML = "text/html"
    JSON = "application/json"
    IMAGE_PNG = "image/png"
    IMAGE_JPEG = "image/jpeg"
    IMAGE_GIF = "image/gif"
    IMAGE_WEBP = "image/webp"
    IMAGE_SVG = "image/svg+xml"
    PDF = "application/pdf"
    AUDIO_MPEG = "audio/mpeg"
    AUDIO_WAV = "audio/wav"
    VIDEO_MP4 = "video/mp4"
    WIDGET = "application/vnd.mcp.widget+json"
    BINARY = "application/octet-stream"


class ToolContent(BaseModel):
    """
    A single content item in a tool result.

    MCP tools can return multiple content items of different types
    in a single response (e.g., text explanation + chart image).

    Design Pattern: Adapter for multi-modal content normalization
    """

    content_type: str = Field(..., description="MIME type of the content")
    data: Any = Field(..., description="Content data (string for text, base64 for binary)")
    encoding: str | None = Field(None, description="Encoding type (e.g., 'base64' for binary)")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (alt_text, dimensions, title, etc.)"
    )

    @property
    def is_text(self) -> bool:
        """Check if content is text-based."""
        return self.content_type.startswith("text/")

    @property
    def is_image(self) -> bool:
        """Check if content is an image."""
        return self.content_type.startswith("image/")

    @property
    def is_widget(self) -> bool:
        """Check if content is an interactive widget."""
        return self.content_type == ContentType.WIDGET.value

    @classmethod
    def text(cls, text: str, content_type: str = "text/plain") -> "ToolContent":
        """Create text content."""
        return cls(content_type=content_type, data=text)

    @classmethod
    def markdown(cls, text: str) -> "ToolContent":
        """Create markdown content."""
        return cls(content_type=ContentType.MARKDOWN.value, data=text)

    @classmethod
    def image(
        cls, base64_data: str, mime_type: str = "image/png", alt_text: str = ""
    ) -> "ToolContent":
        """Create image content."""
        return cls(
            content_type=mime_type,
            data=base64_data,
            encoding="base64",
            metadata={"alt_text": alt_text},
        )

    @classmethod
    def widget(cls, widget_spec: dict[str, Any]) -> "ToolContent":
        """Create interactive widget content."""
        return cls(content_type=ContentType.WIDGET.value, data=widget_spec)


class WidgetSpec(BaseModel):
    """Specification for interactive widgets returned by tools."""

    widget_type: str = Field(
        ..., description="Widget type (data_table, chart, form, code_editor, map)"
    )
    title: str = Field("", description="Widget title")
    data: dict[str, Any] = Field(default_factory=dict, description="Widget-specific data")
    actions: list[dict[str, Any]] = Field(
        default_factory=list, description="Interactive actions (buttons, links)"
    )


# =============================================================================
# TOOL EXECUTION
# =============================================================================


class ToolResultStatus(str, Enum):
    """Status of a tool execution."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ToolResult(BaseModel):
    """
    Result from an MCP tool execution.

    Supports multiple content items for rich, multi-modal responses.
    """

    tool_name: str = Field(..., description="Name of the executed tool")
    status: ToolResultStatus = Field(..., description="Execution status")
    contents: list[ToolContent] = Field(
        default_factory=list, description="Multi-modal content items"
    )
    error: str | None = Field(None, description="Error message if status is error")
    duration_ms: float = Field(0, description="Execution duration in milliseconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @property
    def text_contents(self) -> list[ToolContent]:
        """Get all text content items."""
        return [c for c in self.contents if c.is_text]

    @property
    def image_contents(self) -> list[ToolContent]:
        """Get all image content items."""
        return [c for c in self.contents if c.is_image]

    @property
    def combined_text(self) -> str:
        """Combine all text contents into single string."""
        return "\n".join(c.data for c in self.text_contents if isinstance(c.data, str))

    @property
    def has_images(self) -> bool:
        """Check if result contains images."""
        return any(c.is_image for c in self.contents)

    @property
    def has_widgets(self) -> bool:
        """Check if result contains widgets."""
        return any(c.is_widget for c in self.contents)

    @classmethod
    def success(cls, tool_name: str, contents: list[ToolContent], **kwargs: Any) -> "ToolResult":
        """Create successful result."""
        return cls(tool_name=tool_name, status=ToolResultStatus.SUCCESS, contents=contents, **kwargs)

    @classmethod
    def error(cls, tool_name: str, error: str, **kwargs: Any) -> "ToolResult":
        """Create error result."""
        return cls(tool_name=tool_name, status=ToolResultStatus.ERROR, error=error, **kwargs)


class ToolCall(BaseModel):
    """Request to call a tool."""

    tool_name: str = Field(..., description="Name of tool to call")
    params: dict[str, Any] = Field(default_factory=dict, description="Tool parameters")


# =============================================================================
# ELICITATION (User Input Requests)
# =============================================================================


class ElicitationRequest(BaseModel):
    """Request for user input during tool execution."""

    request_id: str = Field(..., description="Unique request identifier")
    prompt: str = Field(..., description="Question to ask the user")
    input_type: str = Field("text", description="Expected input type (text, choice, confirm)")
    options: list[str] | None = Field(None, description="Options for choice input")
    default: str | None = Field(None, description="Default value")
    timeout_seconds: float = Field(60.0, description="Timeout for response")


class ElicitationResponse(BaseModel):
    """User response to an elicitation request."""

    request_id: str = Field(..., description="Request identifier being responded to")
    value: Any = Field(..., description="User's response value")
    cancelled: bool = Field(False, description="Whether user cancelled the request")
