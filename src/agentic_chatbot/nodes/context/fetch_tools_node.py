"""Fetch tools node for loading MCP tool summaries."""

from typing import Any

from agentic_chatbot.nodes.base import AsyncBaseNode
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class FetchToolsNode(AsyncBaseNode):
    """
    Load MCP tool summaries for supervisor context.

    Type: Context Node

    Fetches tool information from the MCP registry
    so the supervisor knows what tools are available.
    """

    node_name = "fetch_tools"
    description = "Fetch available tool summaries"

    async def prep_async(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Get MCP registry reference."""
        mcp = shared.get("mcp", {})
        return {
            "server_registry": mcp.get("server_registry"),
        }

    async def exec_async(self, prep_res: dict[str, Any]) -> dict[str, Any]:
        """Fetch tool summaries."""
        registry = prep_res.get("server_registry")

        if registry is None:
            logger.warning("No MCP server registry available")
            return {"tool_summaries": [], "tool_count": 0}

        try:
            summaries = await registry.get_all_tool_summaries()
            return {
                "tool_summaries": summaries,
                "tool_count": len(summaries),
            }
        except Exception as e:
            logger.error(f"Failed to fetch tools: {e}")
            return {"tool_summaries": [], "tool_count": 0, "error": str(e)}

    async def post_async(
        self,
        shared: dict[str, Any],
        prep_res: dict[str, Any],
        exec_res: dict[str, Any],
    ) -> str | None:
        """Store tool summaries."""
        shared.setdefault("mcp", {})
        shared["mcp"]["tool_summaries"] = exec_res["tool_summaries"]

        logger.debug(
            "Tools fetched",
            count=exec_res["tool_count"],
        )

        return "default"
