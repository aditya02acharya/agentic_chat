"""Capabilities discovery tool."""

from typing import Any

from agentic_chatbot.mcp.models import (
    MessagingCapabilities,
    OutputDataType,
    ToolResult,
)
from agentic_chatbot.tools.base import LocalTool, LocalToolContext
from agentic_chatbot.tools.registry import LocalToolRegistry


@LocalToolRegistry.register
class CapabilitiesTool(LocalTool):
    """
    Detailed capabilities discovery for the assistant.

    Returns information about:
    - What types of tasks the assistant can handle
    - What tools and operators are available
    - Current limitations and best use cases

    Use this when users ask:
    - "What can you do?"
    - "Can you help with X?"
    - "What are your limitations?"
    """

    name = "capabilities"
    description = "Get detailed information about what this assistant can and cannot do"

    input_schema = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["all", "can_do", "limitations", "best_for"],
                "description": "Which category of capabilities to return",
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

    # Needs access to registries for dynamic capability discovery
    needs_introspection = True

    async def execute(self, context: LocalToolContext) -> ToolResult:
        """Return capabilities based on category."""
        category = context.params.get("category", "all")

        # Core capabilities
        can_do = [
            {
                "category": "Research & Search",
                "capabilities": [
                    "Search the web for current information",
                    "Search internal documents (RAG)",
                    "Synthesize information from multiple sources",
                    "Provide citations for sources",
                ],
            },
            {
                "category": "Analysis",
                "capabilities": [
                    "Analyze data and provide insights",
                    "Compare and contrast information",
                    "Summarize long content",
                    "Extract key points",
                ],
            },
            {
                "category": "Code & Technical",
                "capabilities": [
                    "Generate code in multiple languages",
                    "Explain code and technical concepts",
                    "Debug and troubleshoot issues",
                    "Suggest best practices",
                ],
            },
            {
                "category": "Workflows",
                "capabilities": [
                    "Execute multi-step tasks",
                    "Run parallel operations when possible",
                    "Handle dependencies between steps",
                    "Track progress and report status",
                ],
            },
            {
                "category": "Interaction",
                "capabilities": [
                    "Ask clarifying questions when needed",
                    "Request user input during workflows",
                    "Display rich content (widgets, charts)",
                    "Stream responses in real-time",
                ],
            },
        ]

        limitations = [
            "Cannot access private systems without proper tools configured",
            "Cannot perform actions outside configured tool capabilities",
            "Cannot remember conversations beyond the current session (unless persistence enabled)",
            "Cannot access real-time data unless appropriate tools are available",
            "Cannot execute arbitrary code on the host system",
            "Knowledge cutoff applies to base knowledge (use search for current info)",
        ]

        best_for = [
            "Research tasks requiring multiple sources",
            "Complex multi-step workflows",
            "Data analysis and synthesis",
            "Code generation and explanation",
            "Tasks requiring real-time information (via web search)",
            "Interactive tasks needing user input",
        ]

        # Dynamically discover available tools/operators if registries available
        available_tools = []
        available_operators = []

        if context.mcp_registry:
            try:
                # Get MCP tools
                summaries = await context.mcp_registry.get_all_tool_summaries()
                available_tools.extend([s.name for s in summaries])
            except Exception:
                pass

        if context.local_tool_registry:
            try:
                # Get local tools
                local_summaries = context.local_tool_registry.get_all_summaries()
                available_tools.extend([s.name for s in local_summaries])
            except Exception:
                pass

        if context.operator_registry:
            try:
                # Get operators
                available_operators = context.operator_registry.list_operators()
            except Exception:
                pass

        # Build response based on category
        if category == "can_do":
            content = {"can_do": can_do}
        elif category == "limitations":
            content = {"limitations": limitations}
        elif category == "best_for":
            content = {"best_for": best_for}
        else:  # "all"
            content = {
                "can_do": can_do,
                "limitations": limitations,
                "best_for": best_for,
            }
            if available_tools:
                content["available_tools"] = available_tools
            if available_operators:
                content["available_operators"] = available_operators

        return self.success(
            f"local:{self.name}",
            content,
            content_type="application/json",
        )
