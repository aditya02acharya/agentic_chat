"""Registry for local tools with factory pattern."""

from typing import Any, Type
import time

from agentic_chatbot.mcp.models import (
    ToolSummary,
    ToolSchema,
    ToolResult,
    ToolResultStatus,
    MessagingCapabilities,
    OutputDataType,
)
from agentic_chatbot.tools.base import LocalTool, LocalToolContext
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class LocalToolRegistry:
    """
    Registry for local tools.

    Design Pattern: Registry + Factory

    Similar to OperatorRegistry but for local tools. Provides:
    - Registration via decorator
    - Factory method for instantiation
    - Unified interface matching MCPServerRegistry

    Usage:
        @LocalToolRegistry.register
        class MyTool(LocalTool):
            name = "my_tool"
            description = "Does something useful"
            ...

        # Later
        result = await LocalToolRegistry.execute("my_tool", {"param": "value"})
    """

    _tools: dict[str, Type[LocalTool]] = {}
    _instances: dict[str, LocalTool] = {}  # Cached instances

    @classmethod
    def register(cls, tool_class: Type[LocalTool]) -> Type[LocalTool]:
        """
        Register a local tool class.

        Can be used as a decorator:
            @LocalToolRegistry.register
            class MyTool(LocalTool):
                ...
        """
        name = tool_class.name
        if name in cls._tools:
            logger.warning(f"Overwriting existing local tool: {name}")
        cls._tools[name] = tool_class
        logger.debug(f"Registered local tool: {name}")
        return tool_class

    @classmethod
    def get(cls, name: str) -> Type[LocalTool] | None:
        """Get a tool class by name."""
        # Handle "local:" prefix if present
        if name.startswith("local:"):
            name = name[6:]
        return cls._tools.get(name)

    @classmethod
    def create(cls, name: str) -> LocalTool:
        """
        Create or get cached instance of a tool.

        Args:
            name: Tool name (with or without "local:" prefix)

        Returns:
            LocalTool instance

        Raises:
            KeyError: If tool not registered
        """
        if name.startswith("local:"):
            name = name[6:]

        if name not in cls._instances:
            tool_class = cls._tools.get(name)
            if not tool_class:
                raise KeyError(f"Local tool not registered: {name}")
            cls._instances[name] = tool_class()

        return cls._instances[name]

    @classmethod
    def list_tools(cls) -> list[str]:
        """Get list of registered tool names."""
        return list(cls._tools.keys())

    @classmethod
    def get_all_summaries(cls) -> list[ToolSummary]:
        """Get summaries of all registered local tools."""
        summaries = []
        for name, tool_class in cls._tools.items():
            try:
                # Create temporary instance for summary
                tool = cls.create(name)
                summaries.append(tool.get_summary())
            except Exception as e:
                logger.warning(f"Failed to get summary for {name}: {e}")
        return summaries

    @classmethod
    def get_tool_summary(cls, name: str) -> ToolSummary | None:
        """Get summary for a specific tool."""
        try:
            tool = cls.create(name)
            return tool.get_summary()
        except KeyError:
            return None

    @classmethod
    def get_tool_schema(cls, name: str) -> ToolSchema | None:
        """Get full schema for a specific tool."""
        try:
            tool = cls.create(name)
            return tool.get_schema()
        except KeyError:
            return None

    @classmethod
    async def execute(
        cls,
        name: str,
        params: dict[str, Any] | None = None,
        context: LocalToolContext | None = None,
    ) -> ToolResult:
        """
        Execute a local tool.

        Args:
            name: Tool name (with or without "local:" prefix)
            params: Parameters to pass to the tool
            context: Optional pre-built context

        Returns:
            ToolResult from the tool execution
        """
        if name.startswith("local:"):
            name = name[6:]

        start_time = time.time()

        try:
            tool = cls.create(name)

            # Build context if not provided
            if context is None:
                context = LocalToolContext(params=params or {})
            elif params:
                context.params = params

            # Execute
            result = await tool.execute(context)

            # Update duration
            result.duration_ms = (time.time() - start_time) * 1000

            logger.debug(
                "Local tool executed",
                tool=name,
                duration_ms=result.duration_ms,
                status=result.status.value,
            )

            return result

        except KeyError:
            return ToolResult(
                tool_name=f"local:{name}",
                status=ToolResultStatus.ERROR,
                contents=[],
                error=f"Local tool not found: {name}",
                duration_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error(f"Local tool execution failed: {e}", exc_info=True)
            return ToolResult(
                tool_name=f"local:{name}",
                status=ToolResultStatus.ERROR,
                contents=[],
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

    @classmethod
    def is_local_tool(cls, name: str) -> bool:
        """Check if a tool name refers to a local tool."""
        if name.startswith("local:"):
            return True
        return name in cls._tools

    @classmethod
    def get_tools_text(cls) -> str:
        """Get formatted text of all local tools for prompts."""
        lines = []
        for summary in cls.get_all_summaries():
            messaging = summary.messaging
            capabilities = []
            if messaging.supports_progress:
                capabilities.append("progress")
            if messaging.supports_elicitation:
                capabilities.append("elicitation")
            if messaging.supports_direct_response:
                capabilities.append("direct_response")

            output_types = [t.value for t in messaging.output_types]
            cap_str = f" [caps: {', '.join(capabilities)}]" if capabilities else ""
            out_str = f" [outputs: {', '.join(output_types)}]"

            lines.append(f"- {summary.name}: {summary.description}{out_str}{cap_str}")
        return "\n".join(lines) if lines else "No local tools available"

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tools (useful for testing)."""
        cls._tools.clear()
        cls._instances.clear()
