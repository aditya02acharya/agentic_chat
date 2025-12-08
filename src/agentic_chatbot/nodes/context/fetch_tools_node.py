"""Fetch tools node for loading MCP tool summaries."""

from typing import Any

from ..base import AsyncBaseNode
from ...utils.logging import get_logger

logger = get_logger(__name__)


class FetchToolsNode(AsyncBaseNode):
    """
    Fetches available MCP tools for the supervisor.
    """

    name = "fetch_tools"

    async def execute(self, shared: dict[str, Any]) -> str:
        mcp_registry = shared.get("mcp_registry")
        if mcp_registry:
            try:
                await mcp_registry.refresh()
                shared["available_tools"] = [
                    {"name": t.name, "description": t.description}
                    for t in mcp_registry.tools
                ]
                logger.debug(f"Loaded {len(shared['available_tools'])} MCP tools")
            except Exception as e:
                logger.warning(f"Failed to fetch MCP tools: {e}")
                shared["available_tools"] = []
        else:
            shared["available_tools"] = []

        return "supervisor"
