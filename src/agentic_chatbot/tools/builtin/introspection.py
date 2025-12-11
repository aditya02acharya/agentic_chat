"""Introspection tools for listing available tools and operators."""

from typing import Any

from agentic_chatbot.mcp.models import (
    MessagingCapabilities,
    OutputDataType,
    ToolResult,
)
from agentic_chatbot.tools.base import LocalTool, LocalToolContext
from agentic_chatbot.tools.registry import LocalToolRegistry


@LocalToolRegistry.register
class ListToolsTool(LocalTool):
    """
    List all available tools (local and remote).

    Returns detailed information about each tool including:
    - Name and description
    - Input parameters
    - Messaging capabilities (what it can return)

    Use this when users ask:
    - "What tools do you have?"
    - "How can you search?"
    - "What external services can you use?"
    """

    name = "list_tools"
    description = "List all available tools with their descriptions and capabilities"

    input_schema = {
        "type": "object",
        "properties": {
            "filter": {
                "type": "string",
                "enum": ["all", "local", "remote", "widget_capable", "image_capable"],
                "description": "Filter tools by type or capability",
                "default": "all",
            },
            "detailed": {
                "type": "boolean",
                "description": "Include detailed information (input schema, capabilities)",
                "default": False,
            },
        },
    }

    messaging = MessagingCapabilities(
        output_types=[OutputDataType.TEXT, OutputDataType.JSON],
        supports_progress=False,
        supports_elicitation=False,
        supports_direct_response=False,
        supports_streaming=False,
    )

    needs_introspection = True

    async def execute(self, context: LocalToolContext) -> ToolResult:
        """List available tools based on filter."""
        filter_type = context.params.get("filter", "all")
        detailed = context.params.get("detailed", False)

        local_tools = []
        remote_tools = []

        # Get local tools
        if context.local_tool_registry:
            for summary in context.local_tool_registry.get_all_summaries():
                tool_info = {
                    "name": summary.name,
                    "description": summary.description,
                    "type": "local",
                }
                if detailed:
                    tool_info["messaging"] = {
                        "output_types": [t.value for t in summary.messaging.output_types],
                        "supports_progress": summary.messaging.supports_progress,
                        "supports_direct_response": summary.messaging.supports_direct_response,
                        "supports_elicitation": summary.messaging.supports_elicitation,
                    }
                local_tools.append(tool_info)

        # Get remote (MCP) tools
        if context.mcp_registry:
            try:
                summaries = await context.mcp_registry.get_all_tool_summaries()
                for summary in summaries:
                    tool_info = {
                        "name": summary.name,
                        "description": summary.description,
                        "type": "remote",
                        "server": summary.server_id,
                    }
                    if detailed:
                        tool_info["messaging"] = {
                            "output_types": [t.value for t in summary.messaging.output_types],
                            "supports_progress": summary.messaging.supports_progress,
                            "supports_direct_response": summary.messaging.supports_direct_response,
                            "supports_elicitation": summary.messaging.supports_elicitation,
                        }
                    remote_tools.append(tool_info)
            except Exception as e:
                remote_tools.append({"error": f"Failed to fetch remote tools: {e}"})

        # Apply filter
        if filter_type == "local":
            tools = local_tools
        elif filter_type == "remote":
            tools = remote_tools
        elif filter_type == "widget_capable":
            tools = [
                t for t in (local_tools + remote_tools)
                if t.get("messaging", {}).get("supports_direct_response")
                and "widget" in str(t.get("messaging", {}).get("output_types", []))
            ]
        elif filter_type == "image_capable":
            tools = [
                t for t in (local_tools + remote_tools)
                if "image" in str(t.get("messaging", {}).get("output_types", []))
            ]
        else:  # "all"
            tools = local_tools + remote_tools

        content = {
            "tools": tools,
            "count": len(tools),
            "local_count": len(local_tools),
            "remote_count": len(remote_tools),
        }

        return self.success(
            f"local:{self.name}",
            content,
            content_type="application/json",
        )


@LocalToolRegistry.register
class ListOperatorsTool(LocalTool):
    """
    List all available operators.

    Operators are execution strategies used by the supervisor to
    accomplish tasks. This tool lists what operators are available
    and their capabilities.

    Use this when users ask:
    - "How do you process requests?"
    - "What operators do you have?"
    - "What can you do internally?"
    """

    name = "list_operators"
    description = "List all available operators (execution strategies) and their capabilities"

    input_schema = {
        "type": "object",
        "properties": {
            "filter": {
                "type": "string",
                "enum": ["all", "llm", "mcp", "hybrid"],
                "description": "Filter operators by type",
                "default": "all",
            },
            "detailed": {
                "type": "boolean",
                "description": "Include detailed information (tools used, capabilities)",
                "default": False,
            },
        },
    }

    messaging = MessagingCapabilities(
        output_types=[OutputDataType.TEXT, OutputDataType.JSON],
        supports_progress=False,
        supports_elicitation=False,
        supports_direct_response=False,
        supports_streaming=False,
    )

    needs_introspection = True

    async def execute(self, context: LocalToolContext) -> ToolResult:
        """List available operators."""
        filter_type = context.params.get("filter", "all")
        detailed = context.params.get("detailed", False)

        operators = []

        if context.operator_registry:
            try:
                summaries = context.operator_registry.get_all_summaries()
                for summary in summaries:
                    op_type = summary.get("type", "unknown")

                    # Apply filter
                    if filter_type != "all":
                        if filter_type == "llm" and op_type != "pure_llm":
                            continue
                        elif filter_type == "mcp" and op_type != "mcp_backed":
                            continue
                        elif filter_type == "hybrid" and op_type != "hybrid":
                            continue

                    op_info = {
                        "name": summary["name"],
                        "description": summary.get("description", ""),
                        "type": op_type,
                    }

                    if detailed and "messaging" in summary:
                        op_info["messaging"] = summary["messaging"]

                    operators.append(op_info)
            except Exception as e:
                operators.append({"error": f"Failed to list operators: {e}"})

        content = {
            "operators": operators,
            "count": len(operators),
        }

        return self.success(
            f"local:{self.name}",
            content,
            content_type="application/json",
        )
