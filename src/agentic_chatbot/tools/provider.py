"""Unified tool provider for local and remote tools.

Provides a single interface for the supervisor to access both:
- Local tools (in-process, zero latency)
- Remote MCP tools (network calls to external servers)

The supervisor doesn't need to know which type a tool is - it just
calls tools by name and the provider routes appropriately.
"""

from typing import Any, TYPE_CHECKING

from agentic_chatbot.mcp.models import (
    ToolSummary,
    ToolSchema,
    ToolResult,
    ToolResultStatus,
    MessagingCapabilities,
    OutputDataType,
)
from agentic_chatbot.tools.base import LocalToolContext
from agentic_chatbot.tools.registry import LocalToolRegistry
from agentic_chatbot.utils.logging import get_logger

if TYPE_CHECKING:
    from agentic_chatbot.mcp.registry import MCPServerRegistry
    from agentic_chatbot.mcp.session import MCPSession
    from agentic_chatbot.operators.registry import OperatorRegistry


logger = get_logger(__name__)


class UnifiedToolProvider:
    """
    Unified interface for accessing both local and remote tools.

    Provides:
    - Single method to get all tool summaries (for supervisor context)
    - Automatic routing of tool calls to local or remote execution
    - Consistent ToolResult format regardless of tool type

    Usage:
        provider = UnifiedToolProvider(
            local_registry=LocalToolRegistry,
            mcp_registry=mcp_registry,
        )

        # Get all tools for supervisor prompt
        summaries = await provider.get_all_summaries()

        # Execute a tool (routes automatically)
        result = await provider.execute("local:self_info", {})
        result = await provider.execute("web_search", {"query": "..."})
    """

    def __init__(
        self,
        local_registry: type[LocalToolRegistry] | None = None,
        mcp_registry: "MCPServerRegistry | None" = None,
        operator_registry: "type[OperatorRegistry] | None" = None,
    ):
        """
        Initialize the unified tool provider.

        Args:
            local_registry: LocalToolRegistry class (not instance)
            mcp_registry: MCPServerRegistry instance for remote tools
            operator_registry: OperatorRegistry class for introspection tools
        """
        self._local = local_registry or LocalToolRegistry
        self._mcp = mcp_registry
        self._operator_registry = operator_registry

    @property
    def has_local(self) -> bool:
        """Check if local tools are available."""
        return self._local is not None

    @property
    def has_remote(self) -> bool:
        """Check if remote MCP tools are available."""
        return self._mcp is not None

    async def get_all_summaries(self) -> list[ToolSummary]:
        """
        Get summaries of all available tools (local and remote).

        Returns:
            Combined list of tool summaries
        """
        summaries = []

        # Get local tool summaries (fast, no network)
        if self._local:
            summaries.extend(self._local.get_all_summaries())

        # Get remote tool summaries
        if self._mcp:
            try:
                remote_summaries = await self._mcp.get_all_tool_summaries()
                summaries.extend(remote_summaries)
            except Exception as e:
                logger.warning(f"Failed to get remote tool summaries: {e}")

        return summaries

    def get_all_summaries_sync(self) -> list[ToolSummary]:
        """
        Get summaries of local tools only (synchronous version).

        Useful when you need summaries without async context.
        Only returns local tools - use get_all_summaries() for all tools.
        """
        if self._local:
            return self._local.get_all_summaries()
        return []

    async def get_tool_summary(self, name: str) -> ToolSummary | None:
        """Get summary for a specific tool."""
        if self.is_local_tool(name):
            return self._local.get_tool_summary(name)
        elif self._mcp:
            try:
                return await self._mcp.get_tool_summary(name)
            except Exception:
                return None
        return None

    async def get_tool_schema(self, name: str) -> ToolSchema | None:
        """Get full schema for a specific tool."""
        if self.is_local_tool(name):
            return self._local.get_tool_schema(name)
        elif self._mcp:
            try:
                return await self._mcp.get_tool_schema(name)
            except Exception:
                return None
        return None

    def is_local_tool(self, name: str) -> bool:
        """Check if a tool name refers to a local tool."""
        if name.startswith("local:"):
            return True
        if self._local:
            return self._local.is_local_tool(name)
        return False

    async def execute(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        mcp_session: "MCPSession | None" = None,
        request_id: str | None = None,
        conversation_id: str | None = None,
    ) -> ToolResult:
        """
        Execute a tool by name.

        Automatically routes to local or remote execution based on the tool name.

        Args:
            name: Tool name (with or without "local:" prefix)
            params: Parameters to pass to the tool
            mcp_session: MCP session for remote tools
            request_id: Request ID for tracking
            conversation_id: Conversation ID for context

        Returns:
            ToolResult from tool execution
        """
        params = params or {}

        if self.is_local_tool(name):
            return await self._execute_local(
                name,
                params,
                request_id=request_id,
                conversation_id=conversation_id,
            )
        else:
            return await self._execute_remote(
                name,
                params,
                mcp_session=mcp_session,
            )

    async def _execute_local(
        self,
        name: str,
        params: dict[str, Any],
        request_id: str | None = None,
        conversation_id: str | None = None,
    ) -> ToolResult:
        """Execute a local tool."""
        # Build context for local tool
        context = LocalToolContext(
            params=params,
            request_id=request_id,
            conversation_id=conversation_id,
            # Provide registries for introspection tools
            operator_registry=self._operator_registry,
            mcp_registry=self._mcp,
            local_tool_registry=self._local,
        )

        return await self._local.execute(name, params, context)

    async def _execute_remote(
        self,
        name: str,
        params: dict[str, Any],
        mcp_session: "MCPSession | None" = None,
    ) -> ToolResult:
        """Execute a remote MCP tool."""
        if not self._mcp:
            return ToolResult(
                tool_name=name,
                status=ToolResultStatus.ERROR,
                contents=[],
                error="MCP registry not available",
                duration_ms=0.0,
            )

        if not mcp_session:
            return ToolResult(
                tool_name=name,
                status=ToolResultStatus.ERROR,
                contents=[],
                error="MCP session required for remote tools",
                duration_ms=0.0,
            )

        try:
            # Execute via MCP session
            return await mcp_session.call_tool(name, params)
        except Exception as e:
            logger.error(f"Remote tool execution failed: {e}", exc_info=True)
            return ToolResult(
                tool_name=name,
                status=ToolResultStatus.ERROR,
                contents=[],
                error=str(e),
                duration_ms=0.0,
            )

    def get_tools_text(self) -> str:
        """
        Get formatted text of all tools for prompts.

        Synchronous version that only includes local tools.
        For full list, use async get_all_summaries() and format yourself.
        """
        lines = []

        # Local tools
        if self._local:
            local_text = self._local.get_tools_text()
            if local_text and local_text != "No local tools available":
                lines.append("## Local Tools (fast, no network)")
                lines.append(local_text)

        return "\n".join(lines) if lines else "No tools available"

    async def get_tools_text_async(self) -> str:
        """
        Get formatted text of all tools for prompts (async version).

        Includes both local and remote tools.
        """
        lines = []

        # Local tools
        if self._local:
            local_text = self._local.get_tools_text()
            if local_text and local_text != "No local tools available":
                lines.append("## Local Tools (fast, no network)")
                lines.append(local_text)

        # Remote tools
        if self._mcp:
            try:
                remote_text = self._mcp.get_tool_summaries_text()
                if remote_text and remote_text != "No tools available":
                    if lines:
                        lines.append("")  # Separator
                    lines.append("## Remote Tools (MCP servers)")
                    lines.append(remote_text)
            except Exception as e:
                lines.append(f"## Remote Tools (error: {e})")

        return "\n".join(lines) if lines else "No tools available"

    async def get_tools_with_capability(
        self,
        supports_direct_response: bool | None = None,
        output_type: OutputDataType | None = None,
    ) -> list[ToolSummary]:
        """
        Get tools matching specified capabilities.

        Args:
            supports_direct_response: Filter by direct response support
            output_type: Filter by output type

        Returns:
            List of matching tool summaries
        """
        all_summaries = await self.get_all_summaries()
        matching = []

        for summary in all_summaries:
            messaging = summary.messaging

            if supports_direct_response is not None:
                if messaging.supports_direct_response != supports_direct_response:
                    continue

            if output_type is not None:
                if output_type not in messaging.output_types:
                    continue

            matching.append(summary)

        return matching
