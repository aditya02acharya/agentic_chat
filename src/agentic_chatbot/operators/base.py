"""Base operator class with Strategy pattern."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..mcp.session import MCPSession
    from .context import OperatorContext, OperatorResult


class OperatorType(Enum):
    """Type of operator based on dependencies."""

    PURE_LLM = "pure_llm"
    MCP_BACKED = "mcp_backed"
    HYBRID = "hybrid"


class BaseOperator(ABC):
    """
    Base class for all operators.

    Design Pattern: Strategy Pattern

    Operators are interchangeable algorithms that the Supervisor
    can invoke. Each operator declares its type and requirements.
    """

    name: str
    description: str
    operator_type: OperatorType

    model: str | None = None
    mcp_tools: list[str] = []
    context_requirements: list[str] = []
    fallback_operator: str | None = None
    max_retries: int = 2

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

    def to_dict(self) -> dict[str, Any]:
        """Convert operator metadata to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.operator_type.value,
            "model": self.model,
            "mcp_tools": self.mcp_tools,
            "context_requirements": self.context_requirements,
        }
