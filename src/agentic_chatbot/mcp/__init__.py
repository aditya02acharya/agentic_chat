"""MCP (Model Context Protocol) integration."""

from agentic_chatbot.mcp.models import (
    MCPServerInfo,
    ToolSummary,
    ToolSchema,
    ToolContent,
    ToolResult,
    ToolResultStatus,
    ContentType,
    WidgetSpec,
    ToolCall,
    ElicitationRequest,
    ElicitationResponse,
    # Messaging capabilities
    MessagingCapabilities,
    OutputDataType,
)
from agentic_chatbot.mcp.callbacks import (
    MCPCallbacks,
    MCPProgressCallback,
    MCPElicitationCallback,
    MCPContentCallback,
    MCPErrorCallback,
    # Concrete implementations
    MCPProgressHandler,
    MCPContentHandler,
    MCPErrorHandler,
    MCPElicitationHandler,
    # Elicitation management
    ElicitationManager,
    PendingElicitation,
    # Factory
    create_mcp_callbacks,
)
from agentic_chatbot.mcp.client import MCPClient
from agentic_chatbot.mcp.manager import MCPClientManager
from agentic_chatbot.mcp.registry import MCPServerRegistry
from agentic_chatbot.mcp.session import MCPSession, MCPSessionManager

__all__ = [
    # Models
    "MCPServerInfo",
    "ToolSummary",
    "ToolSchema",
    "ToolContent",
    "ToolResult",
    "ToolResultStatus",
    "ContentType",
    "WidgetSpec",
    "ToolCall",
    "ElicitationRequest",
    "ElicitationResponse",
    # Messaging capabilities
    "MessagingCapabilities",
    "OutputDataType",
    # Callback Protocols
    "MCPCallbacks",
    "MCPProgressCallback",
    "MCPElicitationCallback",
    "MCPContentCallback",
    "MCPErrorCallback",
    # Callback Implementations
    "MCPProgressHandler",
    "MCPContentHandler",
    "MCPErrorHandler",
    "MCPElicitationHandler",
    # Elicitation Management
    "ElicitationManager",
    "PendingElicitation",
    "create_mcp_callbacks",
    # Client
    "MCPClient",
    # Manager
    "MCPClientManager",
    # Registry
    "MCPServerRegistry",
    # Session
    "MCPSession",
    "MCPSessionManager",
]
