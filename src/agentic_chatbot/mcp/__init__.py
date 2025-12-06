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
)
from agentic_chatbot.mcp.callbacks import (
    MCPCallbacks,
    MCPProgressCallback,
    MCPElicitationCallback,
    MCPContentCallback,
    MCPErrorCallback,
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
    # Callbacks
    "MCPCallbacks",
    "MCPProgressCallback",
    "MCPElicitationCallback",
    "MCPContentCallback",
    "MCPErrorCallback",
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
