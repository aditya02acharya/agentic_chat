"""Hierarchical tool catalog for progressive discovery.

This module implements a scalable tool organization system that supports:
- Categorization of tools into logical groups
- Progressive disclosure (summary -> details -> full schema)
- Efficient scaling from 10 to 1000+ tools

Design Pattern: Composite Pattern for hierarchical tool organization

The supervisor starts with a high-level overview and can drill down:
1. Catalog Overview: Categories with counts and descriptions
2. Category View: Groups within a category, tool summaries
3. Group View: Tools within a group with short descriptions
4. Tool Details: Full schema and parameters for a specific tool
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolCategory(str, Enum):
    """High-level tool categories.

    Categories represent broad functional areas. Each category can contain
    multiple groups of related tools.
    """

    # Core capabilities
    INFORMATION_RETRIEVAL = "information_retrieval"
    CODE_EXECUTION = "code_execution"
    DATA_ANALYSIS = "data_analysis"
    CONTENT_GENERATION = "content_generation"

    # System capabilities
    SYSTEM_INTROSPECTION = "system_introspection"
    DOCUMENT_MANAGEMENT = "document_management"
    WORKFLOW_MANAGEMENT = "workflow_management"

    # External integrations
    EXTERNAL_SERVICES = "external_services"
    COMMUNICATION = "communication"

    # Fallback
    UNCATEGORIZED = "uncategorized"


# Human-readable descriptions for categories
CATEGORY_DESCRIPTIONS: dict[ToolCategory, str] = {
    ToolCategory.INFORMATION_RETRIEVAL: "Search and retrieve information from various sources (web, documents, databases)",
    ToolCategory.CODE_EXECUTION: "Execute code, run scripts, and interact with development environments",
    ToolCategory.DATA_ANALYSIS: "Analyze, transform, and visualize data",
    ToolCategory.CONTENT_GENERATION: "Generate text, images, or other content",
    ToolCategory.SYSTEM_INTROSPECTION: "Inspect system capabilities, list available tools and operators",
    ToolCategory.DOCUMENT_MANAGEMENT: "Load, manage, and query uploaded documents",
    ToolCategory.WORKFLOW_MANAGEMENT: "Create and manage multi-step workflows",
    ToolCategory.EXTERNAL_SERVICES: "Interact with external APIs and services",
    ToolCategory.COMMUNICATION: "Send messages, notifications, or interact with users",
    ToolCategory.UNCATEGORIZED: "Tools that don't fit into other categories",
}


class ToolGroup(BaseModel):
    """A group of related tools within a category.

    Groups provide a second level of organization within categories.
    For example, within INFORMATION_RETRIEVAL:
    - web_search: Tools for searching the web
    - rag_search: Tools for searching internal knowledge bases
    - document_search: Tools for searching uploaded documents
    """

    id: str = Field(..., description="Unique group identifier")
    name: str = Field(..., description="Human-readable group name")
    description: str = Field(..., description="What tools in this group do")
    category: ToolCategory = Field(..., description="Parent category")
    tool_count: int = Field(0, description="Number of tools in this group")

    # For progressive disclosure - just tool names and one-liners
    tool_summaries: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of {name, one_liner} for each tool"
    )


class CatalogEntry(BaseModel):
    """Entry for a tool in the catalog with categorization metadata."""

    name: str = Field(..., description="Tool name")
    one_liner: str = Field(..., description="One-line description (max 80 chars)")
    description: str = Field(..., description="Full description")
    category: ToolCategory = Field(ToolCategory.UNCATEGORIZED)
    group_id: str = Field("default", description="Group within category")

    # Lazy-loaded details
    has_schema: bool = Field(True, description="Whether full schema is available")
    is_local: bool = Field(False, description="Whether this is a local tool")
    server_id: str | None = Field(None, description="MCP server ID if remote")


class CategorySummary(BaseModel):
    """Summary of a category for the catalog overview."""

    category: ToolCategory
    description: str
    tool_count: int
    group_count: int
    groups: list[str] = Field(default_factory=list, description="Group names")

    def format_for_prompt(self) -> str:
        """Format for supervisor prompt."""
        groups_str = ", ".join(self.groups[:3])
        if len(self.groups) > 3:
            groups_str += f", +{len(self.groups) - 3} more"
        return (
            f"- **{self.category.value}** ({self.tool_count} tools): "
            f"{self.description}\n"
            f"  Groups: {groups_str}"
        )


class CatalogOverview(BaseModel):
    """High-level overview of the entire tool catalog.

    This is what the supervisor sees initially - just categories
    with counts, not individual tools.
    """

    total_tools: int = 0
    total_categories: int = 0
    categories: list[CategorySummary] = Field(default_factory=list)

    def format_for_prompt(self) -> str:
        """Format for supervisor system prompt."""
        lines = [
            f"## Tool Catalog ({self.total_tools} tools across {self.total_categories} categories)",
            "",
            "Use `browse_tools` to explore categories and discover the right tool.",
            "",
        ]
        for cat in self.categories:
            if cat.tool_count > 0:
                lines.append(cat.format_for_prompt())
        return "\n".join(lines)


class ToolCatalog:
    """
    Central catalog for organizing and discovering tools.

    Implements progressive disclosure:
    1. get_overview() -> High-level categories with counts
    2. get_category(cat) -> Groups and tool summaries in a category
    3. get_group(cat, group) -> Tools in a specific group
    4. get_tool_details(name) -> Full schema for a tool

    Thread-safe and designed for hot-reloading of tool definitions.
    """

    def __init__(self):
        self._entries: dict[str, CatalogEntry] = {}
        self._groups: dict[str, ToolGroup] = {}
        self._tool_schemas: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        category: ToolCategory = ToolCategory.UNCATEGORIZED,
        group_id: str = "default",
        group_name: str | None = None,
        group_description: str | None = None,
        input_schema: dict[str, Any] | None = None,
        is_local: bool = False,
        server_id: str | None = None,
    ) -> None:
        """Register a tool in the catalog."""
        # Create one-liner from description
        one_liner = description.split(".")[0][:80] if description else name

        # Create entry
        entry = CatalogEntry(
            name=name,
            one_liner=one_liner,
            description=description,
            category=category,
            group_id=group_id,
            is_local=is_local,
            server_id=server_id,
            has_schema=input_schema is not None,
        )
        self._entries[name] = entry

        # Store schema if provided
        if input_schema:
            self._tool_schemas[name] = input_schema

        # Ensure group exists
        group_key = f"{category.value}:{group_id}"
        if group_key not in self._groups:
            self._groups[group_key] = ToolGroup(
                id=group_id,
                name=group_name or group_id.replace("_", " ").title(),
                description=group_description or f"Tools for {group_id}",
                category=category,
            )

        # Update group tool count and summaries
        group = self._groups[group_key]
        group.tool_count += 1
        group.tool_summaries.append({
            "name": name,
            "one_liner": one_liner,
        })

    def unregister(self, name: str) -> None:
        """Remove a tool from the catalog."""
        if name in self._entries:
            entry = self._entries[name]
            group_key = f"{entry.category.value}:{entry.group_id}"

            # Update group
            if group_key in self._groups:
                group = self._groups[group_key]
                group.tool_count -= 1
                group.tool_summaries = [
                    s for s in group.tool_summaries if s["name"] != name
                ]
                # Remove empty groups
                if group.tool_count == 0:
                    del self._groups[group_key]

            del self._entries[name]
            self._tool_schemas.pop(name, None)

    def get_overview(self) -> CatalogOverview:
        """Get high-level catalog overview for supervisor prompt.

        Returns categories with tool counts, not individual tools.
        This is the starting point for progressive discovery.
        """
        # Aggregate by category
        category_data: dict[ToolCategory, dict] = {}

        for entry in self._entries.values():
            cat = entry.category
            if cat not in category_data:
                category_data[cat] = {
                    "tool_count": 0,
                    "groups": set(),
                }
            category_data[cat]["tool_count"] += 1
            category_data[cat]["groups"].add(entry.group_id)

        # Build summaries
        summaries = []
        for cat, data in category_data.items():
            summaries.append(CategorySummary(
                category=cat,
                description=CATEGORY_DESCRIPTIONS.get(cat, ""),
                tool_count=data["tool_count"],
                group_count=len(data["groups"]),
                groups=sorted(data["groups"]),
            ))

        # Sort by tool count descending
        summaries.sort(key=lambda s: s.tool_count, reverse=True)

        return CatalogOverview(
            total_tools=len(self._entries),
            total_categories=len(category_data),
            categories=summaries,
        )

    def get_category_details(self, category: ToolCategory) -> dict[str, Any]:
        """Get detailed view of a category including groups and tool summaries."""
        groups = []
        for group_key, group in self._groups.items():
            if group.category == category:
                groups.append({
                    "id": group.id,
                    "name": group.name,
                    "description": group.description,
                    "tool_count": group.tool_count,
                    "tools": group.tool_summaries,
                })

        return {
            "category": category.value,
            "description": CATEGORY_DESCRIPTIONS.get(category, ""),
            "groups": groups,
            "total_tools": sum(g["tool_count"] for g in groups),
        }

    def get_group_details(self, category: ToolCategory, group_id: str) -> dict[str, Any]:
        """Get detailed view of a specific group with all tools."""
        group_key = f"{category.value}:{group_id}"
        if group_key not in self._groups:
            return {"error": f"Group not found: {group_id} in {category.value}"}

        group = self._groups[group_key]

        # Get full tool info for this group
        tools = []
        for entry in self._entries.values():
            if entry.category == category and entry.group_id == group_id:
                tools.append({
                    "name": entry.name,
                    "description": entry.description,
                    "is_local": entry.is_local,
                    "has_schema": entry.has_schema,
                })

        return {
            "group_id": group_id,
            "name": group.name,
            "description": group.description,
            "category": category.value,
            "tools": tools,
        }

    def get_tool_details(self, name: str) -> dict[str, Any]:
        """Get full details for a specific tool including input schema."""
        if name not in self._entries:
            return {"error": f"Tool not found: {name}"}

        entry = self._entries[name]

        return {
            "name": entry.name,
            "description": entry.description,
            "category": entry.category.value,
            "group": entry.group_id,
            "is_local": entry.is_local,
            "server_id": entry.server_id,
            "input_schema": self._tool_schemas.get(name, {}),
        }

    def search_tools(
        self,
        query: str,
        category: ToolCategory | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search tools by name or description."""
        query_lower = query.lower()
        results = []

        for entry in self._entries.values():
            # Apply category filter
            if category and entry.category != category:
                continue

            # Simple text matching
            score = 0
            if query_lower in entry.name.lower():
                score += 2
            if query_lower in entry.description.lower():
                score += 1
            if query_lower in entry.one_liner.lower():
                score += 1

            if score > 0:
                results.append({
                    "name": entry.name,
                    "one_liner": entry.one_liner,
                    "category": entry.category.value,
                    "group": entry.group_id,
                    "score": score,
                })

        # Sort by score and limit
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]

    def get_tools_for_prompt(self, max_tools: int = 20) -> str:
        """Get a concise tool list for prompts (fallback for small catalogs)."""
        if len(self._entries) <= max_tools:
            # Small catalog - show all tools
            lines = []
            for entry in sorted(self._entries.values(), key=lambda e: e.name):
                lines.append(f"- {entry.name}: {entry.one_liner}")
            return "\n".join(lines)
        else:
            # Large catalog - show overview
            return self.get_overview().format_for_prompt()


# Global catalog instance
_catalog: ToolCatalog | None = None


def get_catalog() -> ToolCatalog:
    """Get the global tool catalog instance."""
    global _catalog
    if _catalog is None:
        _catalog = ToolCatalog()
    return _catalog


def reset_catalog() -> None:
    """Reset the global catalog (for testing)."""
    global _catalog
    _catalog = None
