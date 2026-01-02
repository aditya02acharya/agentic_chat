"""Introspection tools for progressive tool discovery.

This module implements hierarchical tool discovery following the
progressive disclosure pattern:

1. browse_tools: Get catalog overview (categories + counts)
2. explore_category: See groups and tool summaries in a category
3. get_tool_info: Get full details for a specific tool
4. search_tools: Search across all tools by keyword

This design scales from 10 to 1000+ tools while keeping the
supervisor's context manageable.
"""

from typing import Any

from agentic_chatbot.mcp.models import (
    MessagingCapabilities,
    OutputDataType,
    ToolResult,
)
from agentic_chatbot.tools.base import LocalTool, LocalToolContext
from agentic_chatbot.tools.registry import LocalToolRegistry
from agentic_chatbot.tools.catalog import (
    ToolCatalog,
    ToolCategory,
    get_catalog,
    CATEGORY_DESCRIPTIONS,
)


@LocalToolRegistry.register
class BrowseToolsTool(LocalTool):
    """
    Browse the tool catalog at various levels of detail.

    This is the ENTRY POINT for tool discovery. Use it to:
    - See all categories and their tool counts (level: "overview")
    - Explore a specific category's groups (level: "category")
    - View tools in a specific group (level: "group")

    The supervisor should start with "overview" and drill down as needed.

    Examples:
    - {"level": "overview"} -> See all categories
    - {"level": "category", "category": "information_retrieval"} -> See groups
    - {"level": "group", "category": "information_retrieval", "group": "web_search"} -> See tools
    """

    name = "browse_tools"
    description = "Browse available tools at different levels of detail (overview -> category -> group)"

    input_schema = {
        "type": "object",
        "properties": {
            "level": {
                "type": "string",
                "enum": ["overview", "category", "group"],
                "description": "Detail level: overview (categories), category (groups), or group (tools)",
                "default": "overview",
            },
            "category": {
                "type": "string",
                "description": "Category to explore (required for 'category' and 'group' levels)",
                "enum": [c.value for c in ToolCategory],
            },
            "group": {
                "type": "string",
                "description": "Group to explore (required for 'group' level)",
            },
        },
        "required": [],
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
        """Browse tools at the specified level."""
        level = context.params.get("level", "overview")
        category_str = context.params.get("category")
        group_id = context.params.get("group")

        # Get or build catalog
        catalog = await self._get_catalog(context)

        if level == "overview":
            return self._browse_overview(catalog)
        elif level == "category":
            if not category_str:
                return self.error(
                    f"local:{self.name}",
                    "Parameter 'category' is required for level='category'"
                )
            try:
                category = ToolCategory(category_str)
            except ValueError:
                valid = [c.value for c in ToolCategory]
                return self.error(
                    f"local:{self.name}",
                    f"Invalid category: {category_str}. Valid: {valid}"
                )
            return self._browse_category(catalog, category)
        elif level == "group":
            if not category_str or not group_id:
                return self.error(
                    f"local:{self.name}",
                    "Parameters 'category' and 'group' are required for level='group'"
                )
            try:
                category = ToolCategory(category_str)
            except ValueError:
                return self.error(
                    f"local:{self.name}",
                    f"Invalid category: {category_str}"
                )
            return self._browse_group(catalog, category, group_id)
        else:
            return self.error(
                f"local:{self.name}",
                f"Invalid level: {level}. Use 'overview', 'category', or 'group'"
            )

    async def _get_catalog(self, context: LocalToolContext) -> ToolCatalog:
        """Get or build the tool catalog from registries."""
        catalog = get_catalog()

        # Populate from local tools if empty
        if not catalog._entries and context.local_tool_registry:
            for summary in context.local_tool_registry.get_all_summaries():
                # Determine category based on tool name/description
                category = self._categorize_tool(summary.name, summary.description)
                catalog.register(
                    name=summary.name,
                    description=summary.description,
                    category=category,
                    group_id=self._get_group_id(summary.name, category),
                    is_local=True,
                )

        # Populate from operators
        if context.operator_registry:
            for summary in context.operator_registry.get_all_summaries():
                category = self._categorize_tool(summary["name"], summary.get("description", ""))
                catalog.register(
                    name=summary["name"],
                    description=summary.get("description", ""),
                    category=category,
                    group_id=self._get_group_id(summary["name"], category),
                    is_local=False,
                )

        # Populate from MCP tools
        if context.mcp_registry:
            try:
                summaries = await context.mcp_registry.get_all_tool_summaries()
                for summary in summaries:
                    category = self._categorize_tool(summary.name, summary.description)
                    catalog.register(
                        name=summary.name,
                        description=summary.description,
                        category=category,
                        group_id=self._get_group_id(summary.name, category),
                        is_local=False,
                        server_id=summary.server_id,
                    )
            except Exception:
                pass  # MCP tools optional

        return catalog

    def _categorize_tool(self, name: str, description: str) -> ToolCategory:
        """Determine category based on tool name and description."""
        name_lower = name.lower()
        desc_lower = description.lower()

        # System introspection
        if any(kw in name_lower for kw in ["list", "browse", "self", "info", "capability"]):
            return ToolCategory.SYSTEM_INTROSPECTION

        # Document management
        if any(kw in name_lower for kw in ["document", "load_doc", "file"]):
            return ToolCategory.DOCUMENT_MANAGEMENT

        # Information retrieval
        if any(kw in name_lower or kw in desc_lower for kw in ["search", "retrieve", "fetch", "query", "rag"]):
            return ToolCategory.INFORMATION_RETRIEVAL

        # Code execution
        if any(kw in name_lower or kw in desc_lower for kw in ["code", "execute", "run", "script", "compile"]):
            return ToolCategory.CODE_EXECUTION

        # Data analysis
        if any(kw in name_lower or kw in desc_lower for kw in ["analyze", "data", "chart", "visualize", "stats"]):
            return ToolCategory.DATA_ANALYSIS

        # Content generation
        if any(kw in name_lower or kw in desc_lower for kw in ["generate", "create", "write", "compose"]):
            return ToolCategory.CONTENT_GENERATION

        # Workflow
        if any(kw in name_lower for kw in ["workflow", "pipeline", "orchestrate"]):
            return ToolCategory.WORKFLOW_MANAGEMENT

        return ToolCategory.UNCATEGORIZED

    def _get_group_id(self, name: str, category: ToolCategory) -> str:
        """Determine group within category."""
        name_lower = name.lower()

        if category == ToolCategory.INFORMATION_RETRIEVAL:
            if "web" in name_lower:
                return "web_search"
            elif "rag" in name_lower:
                return "knowledge_base"
            elif "doc" in name_lower:
                return "document_search"
            return "general_search"

        if category == ToolCategory.SYSTEM_INTROSPECTION:
            if "tool" in name_lower:
                return "tool_discovery"
            elif "operator" in name_lower:
                return "operator_info"
            return "system_info"

        if category == ToolCategory.CODE_EXECUTION:
            if "python" in name_lower:
                return "python"
            elif "javascript" in name_lower or "js" in name_lower:
                return "javascript"
            return "general_execution"

        return "default"

    def _browse_overview(self, catalog: ToolCatalog) -> ToolResult:
        """Return catalog overview."""
        overview = catalog.get_overview()

        content = {
            "level": "overview",
            "total_tools": overview.total_tools,
            "total_categories": overview.total_categories,
            "categories": [
                {
                    "id": cat.category.value,
                    "description": cat.description,
                    "tool_count": cat.tool_count,
                    "groups": cat.groups,
                }
                for cat in overview.categories
                if cat.tool_count > 0
            ],
            "hint": "Use browse_tools with level='category' and a category id to see its groups and tools",
        }

        return self.success(
            f"local:{self.name}",
            content,
            content_type="application/json",
        )

    def _browse_category(self, catalog: ToolCatalog, category: ToolCategory) -> ToolResult:
        """Return category details."""
        details = catalog.get_category_details(category)

        content = {
            "level": "category",
            "category": details["category"],
            "description": details["description"],
            "total_tools": details["total_tools"],
            "groups": details["groups"],
            "hint": "Use browse_tools with level='group' to see all tools in a group, or get_tool_info for a specific tool",
        }

        return self.success(
            f"local:{self.name}",
            content,
            content_type="application/json",
        )

    def _browse_group(self, catalog: ToolCatalog, category: ToolCategory, group_id: str) -> ToolResult:
        """Return group details with all tools."""
        details = catalog.get_group_details(category, group_id)

        if "error" in details:
            return self.error(f"local:{self.name}", details["error"])

        content = {
            "level": "group",
            "group_id": details["group_id"],
            "name": details["name"],
            "description": details["description"],
            "category": details["category"],
            "tools": details["tools"],
            "hint": "Use get_tool_info for full details including input parameters",
        }

        return self.success(
            f"local:{self.name}",
            content,
            content_type="application/json",
        )


@LocalToolRegistry.register
class GetToolInfoTool(LocalTool):
    """
    Get complete information about a specific tool.

    This returns the full tool schema including:
    - Description
    - Input parameters and their types
    - Required vs optional parameters
    - Category and group information

    Use this BEFORE calling a tool to understand its parameters.
    """

    name = "get_tool_info"
    description = "Get full details about a specific tool including input schema and parameters"

    input_schema = {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "Name of the tool to get info about",
            },
        },
        "required": ["tool_name"],
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
        """Get full tool information."""
        tool_name = context.params.get("tool_name")
        if not tool_name:
            return self.error(f"local:{self.name}", "Parameter 'tool_name' is required")

        # Try catalog first
        catalog = get_catalog()
        details = catalog.get_tool_details(tool_name)

        if "error" not in details:
            return self.success(
                f"local:{self.name}",
                details,
                content_type="application/json",
            )

        # Fallback: check local tool registry
        if context.local_tool_registry:
            schema = context.local_tool_registry.get_tool_schema(tool_name)
            if schema:
                return self.success(
                    f"local:{self.name}",
                    {
                        "name": schema.name,
                        "description": schema.description,
                        "input_schema": schema.input_schema,
                        "messaging": {
                            "output_types": [t.value for t in schema.messaging.output_types],
                            "supports_direct_response": schema.messaging.supports_direct_response,
                        },
                    },
                    content_type="application/json",
                )

        # Try MCP registry
        if context.mcp_registry:
            try:
                schema = await context.mcp_registry.get_tool_schema(tool_name)
                if schema:
                    return self.success(
                        f"local:{self.name}",
                        {
                            "name": schema.name,
                            "description": schema.description,
                            "server_id": schema.server_id,
                            "input_schema": schema.input_schema,
                            "messaging": {
                                "output_types": [t.value for t in schema.messaging.output_types],
                            },
                        },
                        content_type="application/json",
                    )
            except Exception:
                pass

        return self.error(f"local:{self.name}", f"Tool not found: {tool_name}")


@LocalToolRegistry.register
class SearchToolsTool(LocalTool):
    """
    Search for tools by keyword.

    Use this when you know what you want to do but not which tool does it.
    Searches tool names and descriptions.

    Examples:
    - "search": Find tools related to searching
    - "image": Find tools that work with images
    - "python": Find Python-related tools
    """

    name = "search_tools"
    description = "Search for tools by keyword in name or description"

    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search term to find relevant tools",
            },
            "category": {
                "type": "string",
                "description": "Optional: limit search to a specific category",
                "enum": [c.value for c in ToolCategory],
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default: 10)",
                "default": 10,
            },
        },
        "required": ["query"],
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
        """Search for tools."""
        query = context.params.get("query")
        if not query:
            return self.error(f"local:{self.name}", "Parameter 'query' is required")

        category_str = context.params.get("category")
        limit = context.params.get("limit", 10)

        category = None
        if category_str:
            try:
                category = ToolCategory(category_str)
            except ValueError:
                pass

        catalog = get_catalog()
        results = catalog.search_tools(query, category=category, limit=limit)

        return self.success(
            f"local:{self.name}",
            {
                "query": query,
                "category_filter": category_str,
                "results": results,
                "count": len(results),
                "hint": "Use get_tool_info for full details about a specific tool",
            },
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
