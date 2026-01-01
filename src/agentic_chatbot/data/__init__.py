"""Unified data structures for information flow.

This module defines the core data types used throughout the agentic chatbot:

1. **ContentBlock**: The atomic unit of information (text, image, widget, etc.)
2. **SourcedContent**: ContentBlock with provenance tracking for citations
3. **ExecutionInput**: What operators/tools receive
4. **ExecutionOutput**: What operators/tools return
5. **Directive**: What supervisor delegates to operators

Design Principles:
- Single source of truth for each concept
- Immutable core types (ContentBlock, SourcedContent)
- Clear separation between input, output, and internal types
- Provenance tracking for citations and debugging
"""

from agentic_chatbot.data.content import (
    ContentBlock,
    ContentType,
    TextContent,
    ImageContent,
    WidgetContent,
    JsonContent,
    ErrorContent,
)
from agentic_chatbot.data.sourced import (
    SourcedContent,
    ContentSource,
    ContentSummary,
    create_sourced_content,
)
from agentic_chatbot.data.execution import (
    ExecutionInput,
    ExecutionOutput,
    ExecutionStatus,
    TaskInfo,
)
from agentic_chatbot.data.directive import (
    Directive,
    DirectiveType,
    DirectiveRecord,
    DirectiveOutcome,
)

__all__ = [
    # Core content
    "ContentBlock",
    "ContentType",
    "TextContent",
    "ImageContent",
    "WidgetContent",
    "JsonContent",
    "ErrorContent",
    # Sourced content
    "SourcedContent",
    "ContentSource",
    "ContentSummary",
    "create_sourced_content",
    # Execution
    "ExecutionInput",
    "ExecutionOutput",
    "ExecutionStatus",
    "TaskInfo",
    # Directive
    "Directive",
    "DirectiveType",
    "DirectiveRecord",
    "DirectiveOutcome",
]
