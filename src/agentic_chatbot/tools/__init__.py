"""Local tools module for in-process tool execution.

Local tools complement remote MCP tools by providing:
- Zero network latency for simple operations
- Self-awareness (version, capabilities, release notes)
- Introspection (what tools/operators are available)
- Simple utilities

The supervisor sees both local and remote tools through a unified interface.
"""

from agentic_chatbot.tools.base import LocalTool, LocalToolContext
from agentic_chatbot.tools.registry import LocalToolRegistry
from agentic_chatbot.tools.provider import UnifiedToolProvider
from agentic_chatbot.tools.builtin import register_builtin_tools

__all__ = [
    "LocalTool",
    "LocalToolContext",
    "LocalToolRegistry",
    "UnifiedToolProvider",
    "register_builtin_tools",
]
