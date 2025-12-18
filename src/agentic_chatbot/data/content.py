"""Core content types - the atomic unit of information.

ContentBlock is the fundamental unit of information that flows through the system.
It's immutable and self-describing, containing both data and metadata.

Design Principles:
- Immutable (frozen dataclass)
- Self-describing (content_type field)
- Serializable (to_dict/from_dict)
- Type-safe (specialized subclasses)

Usage:
    # Create text content
    text = TextContent("Hello, world!")

    # Create from any type
    content = ContentBlock.create("markdown", "# Title")

    # Check type
    if content.is_text:
        print(content.as_text)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar
import json


class ContentType(str, Enum):
    """Supported content types."""

    # Text types
    TEXT = "text/plain"
    MARKDOWN = "text/markdown"
    HTML = "text/html"
    CODE = "text/x-code"

    # Data types
    JSON = "application/json"

    # Image types
    IMAGE_PNG = "image/png"
    IMAGE_JPEG = "image/jpeg"
    IMAGE_GIF = "image/gif"
    IMAGE_WEBP = "image/webp"
    IMAGE_SVG = "image/svg+xml"

    # Widget/interactive
    WIDGET = "application/vnd.widget+json"

    # Error/status
    ERROR = "application/vnd.error+json"

    # Binary
    BINARY = "application/octet-stream"

    @classmethod
    def from_string(cls, s: str) -> "ContentType":
        """Get ContentType from string, with fallback."""
        try:
            return cls(s)
        except ValueError:
            # Try matching by prefix
            if s.startswith("text/"):
                return cls.TEXT
            if s.startswith("image/"):
                return cls.IMAGE_PNG
            return cls.BINARY


@dataclass(frozen=True)
class ContentBlock:
    """
    Immutable unit of content.

    The fundamental data type that flows through the system.
    Can represent text, images, widgets, errors, or any other content.

    Attributes:
        content_type: MIME type of the content
        data: The actual content data
        encoding: Optional encoding (e.g., "base64" for binary)
        metadata: Additional metadata (alt_text, language, etc.)
        created_at: Timestamp when content was created
    """

    content_type: ContentType
    data: Any
    encoding: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Type checking properties
    @property
    def is_text(self) -> bool:
        """Check if content is text-based."""
        return self.content_type in (
            ContentType.TEXT,
            ContentType.MARKDOWN,
            ContentType.HTML,
            ContentType.CODE,
        )

    @property
    def is_image(self) -> bool:
        """Check if content is an image."""
        return self.content_type in (
            ContentType.IMAGE_PNG,
            ContentType.IMAGE_JPEG,
            ContentType.IMAGE_GIF,
            ContentType.IMAGE_WEBP,
            ContentType.IMAGE_SVG,
        )

    @property
    def is_widget(self) -> bool:
        """Check if content is an interactive widget."""
        return self.content_type == ContentType.WIDGET

    @property
    def is_error(self) -> bool:
        """Check if content represents an error."""
        return self.content_type == ContentType.ERROR

    @property
    def is_json(self) -> bool:
        """Check if content is JSON data."""
        return self.content_type == ContentType.JSON

    # Data access properties
    @property
    def as_text(self) -> str:
        """Get content as text string."""
        if isinstance(self.data, str):
            return self.data
        if isinstance(self.data, bytes):
            return self.data.decode("utf-8")
        return str(self.data)

    @property
    def as_dict(self) -> dict[str, Any]:
        """Get content as dictionary (for JSON/widget content)."""
        if isinstance(self.data, dict):
            return self.data
        if isinstance(self.data, str):
            try:
                return json.loads(self.data)
            except json.JSONDecodeError:
                return {"text": self.data}
        return {"value": self.data}

    # Serialization
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "content_type": self.content_type.value,
            "data": self.data,
            "encoding": self.encoding,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ContentBlock":
        """Deserialize from dictionary."""
        return cls(
            content_type=ContentType.from_string(d["content_type"]),
            data=d["data"],
            encoding=d.get("encoding"),
            metadata=d.get("metadata", {}),
            created_at=datetime.fromisoformat(d["created_at"])
            if "created_at" in d
            else datetime.utcnow(),
        )

    # Factory methods
    @classmethod
    def create(
        cls,
        content_type: str | ContentType,
        data: Any,
        **kwargs: Any,
    ) -> "ContentBlock":
        """Create ContentBlock from type string and data."""
        if isinstance(content_type, str):
            content_type = ContentType.from_string(content_type)
        return cls(content_type=content_type, data=data, **kwargs)


# =============================================================================
# SPECIALIZED CONTENT TYPES
# =============================================================================


@dataclass(frozen=True)
class TextContent(ContentBlock):
    """Text content with language support."""

    content_type: ContentType = field(default=ContentType.TEXT)
    data: str = ""
    language: str | None = None  # For code: python, javascript, etc.

    def __init__(
        self,
        text: str,
        content_type: ContentType = ContentType.TEXT,
        language: str | None = None,
        **kwargs: Any,
    ):
        # Work around frozen dataclass
        object.__setattr__(self, "content_type", content_type)
        object.__setattr__(self, "data", text)
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "encoding", None)
        object.__setattr__(self, "metadata", kwargs.get("metadata", {}))
        object.__setattr__(
            self, "created_at", kwargs.get("created_at", datetime.utcnow())
        )

    @classmethod
    def plain(cls, text: str) -> "TextContent":
        """Create plain text content."""
        return cls(text, ContentType.TEXT)

    @classmethod
    def markdown(cls, text: str) -> "TextContent":
        """Create markdown content."""
        return cls(text, ContentType.MARKDOWN)

    @classmethod
    def html(cls, text: str) -> "TextContent":
        """Create HTML content."""
        return cls(text, ContentType.HTML)

    @classmethod
    def code(cls, code: str, language: str = "python") -> "TextContent":
        """Create code content."""
        return cls(code, ContentType.CODE, language=language)


@dataclass(frozen=True)
class ImageContent(ContentBlock):
    """Image content with metadata."""

    content_type: ContentType = field(default=ContentType.IMAGE_PNG)
    data: str = ""  # Base64-encoded
    encoding: str = "base64"
    alt_text: str = ""
    width: int | None = None
    height: int | None = None

    def __init__(
        self,
        base64_data: str,
        mime_type: ContentType = ContentType.IMAGE_PNG,
        alt_text: str = "",
        width: int | None = None,
        height: int | None = None,
        **kwargs: Any,
    ):
        object.__setattr__(self, "content_type", mime_type)
        object.__setattr__(self, "data", base64_data)
        object.__setattr__(self, "encoding", "base64")
        object.__setattr__(self, "alt_text", alt_text)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(
            self,
            "metadata",
            {
                "alt_text": alt_text,
                "width": width,
                "height": height,
                **kwargs.get("metadata", {}),
            },
        )
        object.__setattr__(
            self, "created_at", kwargs.get("created_at", datetime.utcnow())
        )

    @classmethod
    def png(cls, base64_data: str, alt_text: str = "") -> "ImageContent":
        """Create PNG image content."""
        return cls(base64_data, ContentType.IMAGE_PNG, alt_text)

    @classmethod
    def jpeg(cls, base64_data: str, alt_text: str = "") -> "ImageContent":
        """Create JPEG image content."""
        return cls(base64_data, ContentType.IMAGE_JPEG, alt_text)

    @classmethod
    def svg(cls, svg_data: str, alt_text: str = "") -> "ImageContent":
        """Create SVG image content (not base64)."""
        content = cls.__new__(cls)
        object.__setattr__(content, "content_type", ContentType.IMAGE_SVG)
        object.__setattr__(content, "data", svg_data)
        object.__setattr__(content, "encoding", None)  # SVG is text
        object.__setattr__(content, "alt_text", alt_text)
        object.__setattr__(content, "width", None)
        object.__setattr__(content, "height", None)
        object.__setattr__(content, "metadata", {"alt_text": alt_text})
        object.__setattr__(content, "created_at", datetime.utcnow())
        return content


@dataclass(frozen=True)
class WidgetContent(ContentBlock):
    """Interactive widget content."""

    content_type: ContentType = field(default=ContentType.WIDGET)
    data: dict[str, Any] = field(default_factory=dict)
    widget_type: str = ""
    title: str = ""

    def __init__(
        self,
        widget_type: str,
        data: dict[str, Any],
        title: str = "",
        actions: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        widget_data = {
            "widget_type": widget_type,
            "title": title,
            "data": data,
            "actions": actions or [],
        }
        object.__setattr__(self, "content_type", ContentType.WIDGET)
        object.__setattr__(self, "data", widget_data)
        object.__setattr__(self, "encoding", None)
        object.__setattr__(self, "widget_type", widget_type)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "metadata", kwargs.get("metadata", {}))
        object.__setattr__(
            self, "created_at", kwargs.get("created_at", datetime.utcnow())
        )

    @classmethod
    def data_table(
        cls,
        columns: list[str],
        rows: list[list[Any]],
        title: str = "",
    ) -> "WidgetContent":
        """Create data table widget."""
        return cls(
            widget_type="data_table",
            data={"columns": columns, "rows": rows},
            title=title,
        )

    @classmethod
    def chart(
        cls,
        chart_type: str,
        data: dict[str, Any],
        title: str = "",
    ) -> "WidgetContent":
        """Create chart widget."""
        return cls(
            widget_type="chart",
            data={"chart_type": chart_type, **data},
            title=title,
        )


@dataclass(frozen=True)
class JsonContent(ContentBlock):
    """Structured JSON content."""

    content_type: ContentType = field(default=ContentType.JSON)
    data: dict[str, Any] = field(default_factory=dict)

    def __init__(self, data: dict[str, Any], **kwargs: Any):
        object.__setattr__(self, "content_type", ContentType.JSON)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "encoding", None)
        object.__setattr__(self, "metadata", kwargs.get("metadata", {}))
        object.__setattr__(
            self, "created_at", kwargs.get("created_at", datetime.utcnow())
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from JSON data."""
        return self.data.get(key, default)


@dataclass(frozen=True)
class ErrorContent(ContentBlock):
    """Error content."""

    content_type: ContentType = field(default=ContentType.ERROR)
    error_type: str = "error"
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        message: str,
        error_type: str = "error",
        details: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        error_data = {
            "error_type": error_type,
            "message": message,
            "details": details or {},
        }
        object.__setattr__(self, "content_type", ContentType.ERROR)
        object.__setattr__(self, "data", error_data)
        object.__setattr__(self, "encoding", None)
        object.__setattr__(self, "error_type", error_type)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "details", details or {})
        object.__setattr__(self, "metadata", kwargs.get("metadata", {}))
        object.__setattr__(
            self, "created_at", kwargs.get("created_at", datetime.utcnow())
        )

    @classmethod
    def timeout(cls, message: str = "Operation timed out") -> "ErrorContent":
        """Create timeout error."""
        return cls(message, error_type="timeout")

    @classmethod
    def validation(cls, message: str, field: str = "") -> "ErrorContent":
        """Create validation error."""
        return cls(message, error_type="validation", details={"field": field})

    @classmethod
    def execution(cls, message: str, details: dict[str, Any] | None = None) -> "ErrorContent":
        """Create execution error."""
        return cls(message, error_type="execution", details=details)
