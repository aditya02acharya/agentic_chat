"""Base operator class and types.

Operators are the units of execution that the supervisor delegates to.
They receive ExecutionInput and return ExecutionOutput.

The atomic unit of information is ContentBlock (from data.content).
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

from agentic_chatbot.mcp.models import MessagingCapabilities, OutputDataType
from agentic_chatbot.data.execution import ExecutionInput, ExecutionOutput
from agentic_chatbot.data.content import ContentBlock, TextContent

if TYPE_CHECKING:
    from agentic_chatbot.mcp.session import MCPSession
    # Legacy types for backward compatibility
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

    Messaging Capabilities:
    - output_types: What data types this operator can return (text, image, widget, etc.)
    - supports_progress: Whether operator can report intermediate progress
    - supports_elicitation: Whether operator can request user input
    - supports_direct_response: Whether operator can bypass writer and send directly to user
    - supports_streaming: Whether operator can stream content incrementally
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

    # Messaging capabilities - subclasses can override these
    output_types: list[OutputDataType] = [OutputDataType.TEXT]
    supports_progress: bool = False
    supports_elicitation: bool = False
    supports_direct_response: bool = False
    supports_streaming: bool = False

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

    async def run(
        self,
        input: ExecutionInput,
        mcp_session: "MCPSession | None" = None,
    ) -> ExecutionOutput:
        """
        Execute the operator with the new unified interface.

        This is the preferred method for new code. Override this method
        in new operators.

        Args:
            input: ExecutionInput with query, task, and messaging capabilities
            mcp_session: MCP session (required if requires_mcp=True)

        Returns:
            ExecutionOutput with contents and metadata
        """
        # Default implementation converts to legacy interface
        # Subclasses should override this method directly
        from agentic_chatbot.operators.context import OperatorContext, OperatorResult

        # Convert ExecutionInput to OperatorContext
        context = OperatorContext(
            query=input.query,
            step_results={
                k: v.text for k, v in input.step_results.items()
            },
            extra=input.extra,
        )

        # Wire up messaging if available
        if input.messaging_enabled:
            from agentic_chatbot.operators.context import MessagingContext

            messaging = MessagingContext(
                emitter=input._emitter,
                elicitation_manager=input._elicitation_manager,
                request_id=input._request_id,
                operator_name=input._operator_name,
            )
            context.set_messaging(messaging)

        # Call legacy execute
        result = await self.execute(context, mcp_session)

        # Convert OperatorResult to ExecutionOutput
        return self._convert_result(result, input)

    def _convert_result(
        self,
        result: "OperatorResult",
        input: ExecutionInput,
    ) -> ExecutionOutput:
        """Convert legacy OperatorResult to ExecutionOutput."""
        from agentic_chatbot.data.execution import ExecutionStatus

        if not result.success:
            return ExecutionOutput.error(
                result.error or "Unknown error",
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )

        # Convert output to ContentBlock
        contents = []
        if result.output:
            if isinstance(result.output, str):
                contents.append(TextContent.markdown(result.output))
            elif isinstance(result.output, dict):
                from agentic_chatbot.data.content import JsonContent

                contents.append(JsonContent(result.output))
            else:
                contents.append(TextContent.plain(str(result.output)))

        # Convert direct responses
        direct_responses = []
        for tc in result.direct_responses:
            direct_responses.append(
                ContentBlock.create(tc.content_type, tc.data)
            )

        return ExecutionOutput(
            status=ExecutionStatus.SUCCESS,
            contents=contents,
            direct_responses=direct_responses,
            sent_direct_response=result.sent_direct_response,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    @abstractmethod
    async def execute(
        self,
        context: "OperatorContext",
        mcp_session: "MCPSession | None" = None,
    ) -> "OperatorResult":
        """
        Execute the operator (legacy interface).

        DEPRECATED: Override run() instead for new operators.

        Args:
            context: Focused context built by ContextAssembler
            mcp_session: MCP session (required if requires_mcp=True)

        Returns:
            OperatorResult with output and metadata
        """
        pass

    async def run_with_fallback(
        self,
        input: ExecutionInput,
        mcp_session: "MCPSession | None" = None,
    ) -> ExecutionOutput:
        """
        Execute with fallback on failure.

        Args:
            input: ExecutionInput
            mcp_session: Optional MCP session

        Returns:
            Result from this operator or fallback
        """
        from agentic_chatbot.core.exceptions import OperatorError
        from agentic_chatbot.operators.registry import OperatorRegistry

        try:
            return await self.run(input, mcp_session)
        except OperatorError as e:
            if self.fallback_operator:
                fallback = OperatorRegistry.create(self.fallback_operator)
                return await fallback.run(input, mcp_session)
            raise

    async def execute_with_fallback(
        self,
        context: "OperatorContext",
        mcp_session: "MCPSession | None" = None,
    ) -> "OperatorResult":
        """
        Execute with fallback on failure (legacy interface).

        DEPRECATED: Use run_with_fallback() instead.

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

    @property
    def messaging_capabilities(self) -> MessagingCapabilities:
        """Get messaging capabilities as a MessagingCapabilities object."""
        return MessagingCapabilities(
            output_types=self.output_types,
            supports_progress=self.supports_progress,
            supports_elicitation=self.supports_elicitation,
            supports_direct_response=self.supports_direct_response,
            supports_streaming=self.supports_streaming,
        )

    @property
    def can_bypass_writer(self) -> bool:
        """Check if operator can send responses directly to user."""
        return self.supports_direct_response

    @property
    def returns_rich_content(self) -> bool:
        """Check if operator can return images, widgets, or other rich content."""
        rich_types = {OutputDataType.IMAGE, OutputDataType.WIDGET, OutputDataType.HTML}
        return bool(rich_types.intersection(set(self.output_types)))

    def get_tool_summary(self) -> dict[str, Any]:
        """Get summary for supervisor context including messaging capabilities."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.operator_type.value,
            "messaging": {
                "output_types": [t.value for t in self.output_types],
                "supports_progress": self.supports_progress,
                "supports_elicitation": self.supports_elicitation,
                "supports_direct_response": self.supports_direct_response,
                "supports_streaming": self.supports_streaming,
            },
        }
