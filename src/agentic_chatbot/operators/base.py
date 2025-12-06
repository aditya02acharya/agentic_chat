"""Base operator class and types."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentic_chatbot.mcp.session import MCPSession
    from agentic_chatbot.operators.context import OperatorContext, OperatorResult


class OperatorType(Enum):
    """Type of operator based on dependencies."""

    PURE_LLM = "pure_llm"  # Only needs LLM
    MCP_BACKED = "mcp_backed"  # Only needs MCP tools
    HYBRID = "hybrid"  # Needs both LLM and MCP


class BaseOperator(ABC):
    """
    Base class for all operators.

    Design Pattern: Strategy Pattern

    Operators are interchangeable algorithms that the Supervisor
    can invoke. Each operator declares its type and requirements.

    Operators come in three types:
    - PURE_LLM: Only call LLM. Fast path, no MCP overhead.
    - MCP_BACKED: Only call MCP tools. Handle session lifecycle internally.
    - HYBRID: Use LLM for reasoning AND MCP tools for execution.
    """

    # Metadata - must be set by subclasses
    name: str
    description: str
    operator_type: OperatorType

    # LLM config (for PURE_LLM and HYBRID)
    model: str | None = None  # "haiku" | "sonnet" | None

    # MCP config (for MCP_BACKED and HYBRID)
    mcp_tools: list[str] = []  # Which MCP tools this operator may use

    # Context requirements (for ContextAssembler)
    context_requirements: list[str] = []

    # Error handling
    fallback_operator: str | None = None  # Operator to use if this fails
    max_retries: int = 2

    def __init__(self, **kwargs: Any):
        """Initialize operator with optional overrides."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @property
    def requires_mcp(self) -> bool:
        """Check if operator needs MCP session."""
        return self.operator_type in (OperatorType.MCP_BACKED, OperatorType.HYBRID)

    @property
    def requires_llm(self) -> bool:
        """Check if operator needs LLM."""
        return self.operator_type in (OperatorType.PURE_LLM, OperatorType.HYBRID)

    @abstractmethod
    async def execute(
        self,
        context: "OperatorContext",
        mcp_session: "MCPSession | None" = None,
    ) -> "OperatorResult":
        """
        Execute the operator.

        Args:
            context: Focused context built by ContextAssembler
            mcp_session: MCP session (required if requires_mcp=True)

        Returns:
            OperatorResult with output and metadata
        """
        pass

    async def execute_with_fallback(
        self,
        context: "OperatorContext",
        mcp_session: "MCPSession | None" = None,
    ) -> "OperatorResult":
        """
        Execute with fallback on failure.

        Design Pattern: Chain of Responsibility (Layer 2 error handling)

        Args:
            context: Operator context
            mcp_session: Optional MCP session

        Returns:
            Result from this operator or fallback
        """
        from agentic_chatbot.core.exceptions import OperatorError
        from agentic_chatbot.operators.registry import OperatorRegistry

        try:
            return await self.execute(context, mcp_session)
        except OperatorError as e:
            if self.fallback_operator:
                fallback = OperatorRegistry.create(self.fallback_operator)
                return await fallback.execute(context, mcp_session)
            raise

    def get_tool_summary(self) -> dict[str, str]:
        """Get summary for supervisor context."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.operator_type.value,
        }
