"""Self-awareness tool for bot information and release notes."""

import json
from pathlib import Path
from typing import Any

from agentic_chatbot.mcp.models import (
    MessagingCapabilities,
    OutputDataType,
    ToolResult,
    ToolContent,
)
from agentic_chatbot.tools.base import LocalTool, LocalToolContext
from agentic_chatbot.tools.registry import LocalToolRegistry


# Version and release info - update this with releases
BOT_INFO = {
    "name": "AgenticBot",
    "version": "1.0.0",
    "description": "A ReACT-based agentic chatbot with multi-step workflow execution",
    "architecture": "LangGraph + MCP",
}

# Release notes - add new entries at the top
RELEASE_NOTES = """
## v1.0.0 (Current)
- Initial release with ReACT-based supervisor
- Multi-step workflow execution with parallel batching
- MCP tool integration for web search, RAG, and more
- Context optimization with data summaries
- Direct response support for widgets and rich content
- User elicitation for interactive workflows

## Features
- **Supervisor**: Central decision-making agent using ReACT pattern
- **Operators**: Pluggable execution strategies (LLM, MCP, Hybrid)
- **Workflows**: Multi-step execution with dependency resolution
- **Streaming**: Real-time SSE events for progress tracking
- **Citations**: Source tracking with GitHub-style footnotes
"""

# High-level capabilities
CAPABILITIES = [
    "Web search and research",
    "Document analysis and RAG (Retrieval-Augmented Generation)",
    "Multi-step workflow execution with parallel processing",
    "Code generation and analysis",
    "Data synthesis with source citations",
    "Interactive user input during workflows",
    "Rich content responses (widgets, images)",
]


@LocalToolRegistry.register
class SelfInfoTool(LocalTool):
    """
    Returns bot information, version, and release notes.

    Use this tool when users ask:
    - "What are you?" / "Who are you?"
    - "What version are you?"
    - "What's new?" / "What are the recent updates?"
    - "Tell me about yourself"
    """

    name = "self_info"
    description = "Get information about this assistant including version, capabilities, and recent updates"

    input_schema = {
        "type": "object",
        "properties": {
            "section": {
                "type": "string",
                "enum": ["all", "version", "capabilities", "release_notes"],
                "description": "Which section of info to return",
                "default": "all",
            }
        },
    }

    messaging = MessagingCapabilities(
        output_types=[OutputDataType.TEXT, OutputDataType.JSON],
        supports_progress=False,
        supports_elicitation=False,
        supports_direct_response=False,
        supports_streaming=False,
    )

    async def execute(self, context: LocalToolContext) -> ToolResult:
        """Return bot information based on requested section."""
        section = context.params.get("section", "all")

        if section == "version":
            content = {
                "name": BOT_INFO["name"],
                "version": BOT_INFO["version"],
            }
        elif section == "capabilities":
            content = {
                "capabilities": CAPABILITIES,
            }
        elif section == "release_notes":
            content = {
                "release_notes": RELEASE_NOTES.strip(),
            }
        else:  # "all"
            content = {
                **BOT_INFO,
                "capabilities": CAPABILITIES,
                "release_notes": RELEASE_NOTES.strip(),
            }

        return self.success(
            f"local:{self.name}",
            content,
            content_type="application/json",
        )


def get_bot_version() -> str:
    """Get current bot version string."""
    return BOT_INFO["version"]


def get_bot_name() -> str:
    """Get bot name."""
    return BOT_INFO["name"]
