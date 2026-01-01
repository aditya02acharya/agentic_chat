"""Execution input and output types.

ExecutionInput: What operators/tools receive
ExecutionOutput: What operators/tools return

These types standardize the interface between the graph and operators/tools,
replacing the previous mix of OperatorContext, OperatorResult, ToolResult, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING

from agentic_chatbot.data.content import ContentBlock, TextContent, ErrorContent
from agentic_chatbot.data.sourced import SourcedContent, ContentSource

if TYPE_CHECKING:
    from agentic_chatbot.events.emitter import EventEmitter
    from agentic_chatbot.mcp.callbacks import ElicitationManager


class ExecutionStatus(str, Enum):
    """Status of an execution."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"  # Waiting for user input


@dataclass
class ExecutionInput:
    """
    Input for operator/tool execution.

    Consolidates:
    - OperatorContext: query, step_results, extra
    - TaskContext: task_description, goal, scope, constraints
    - MessagingContext: emitter, elicitation_manager

    Usage:
        async def execute(self, input: ExecutionInput) -> ExecutionOutput:
            # Access query
            query = input.query

            # Access task delegation info
            if input.task:
                print(f"Goal: {input.task.goal}")

            # Send progress (if messaging enabled)
            await input.send_progress(0.5, "Processing...")

            # Access previous step results
            prev_data = input.get_step_result("search")

            return ExecutionOutput.success(TextContent.markdown("Done!"))
    """

    # Core query/task
    query: str
    task: "TaskInfo | None" = None

    # Previous step results (for workflows)
    step_results: dict[str, "ExecutionOutput"] = field(default_factory=dict)

    # Extra context (operator-specific parameters)
    extra: dict[str, Any] = field(default_factory=dict)

    # Messaging context (optional)
    _emitter: "EventEmitter | None" = field(default=None, repr=False)
    _elicitation_manager: "ElicitationManager | None" = field(default=None, repr=False)
    _request_id: str | None = field(default=None, repr=False)
    _operator_name: str = field(default="unknown", repr=False)

    # Direct responses tracking
    _direct_responses: list[ContentBlock] = field(default_factory=list, repr=False)
    _has_sent_direct: bool = field(default=False, repr=False)

    # Properties
    @property
    def messaging_enabled(self) -> bool:
        """Check if messaging is enabled."""
        return self._emitter is not None

    @property
    def has_direct_responses(self) -> bool:
        """Check if direct responses were sent."""
        return self._has_sent_direct

    @property
    def direct_responses(self) -> list[ContentBlock]:
        """Get direct responses that were sent."""
        return self._direct_responses.copy()

    # Step result access
    def get_step_result(self, step_id: str) -> "ExecutionOutput | None":
        """Get result from a previous workflow step."""
        return self.step_results.get(step_id)

    def get_step_content(self, step_id: str) -> ContentBlock | None:
        """Get primary content from a previous step."""
        result = self.get_step_result(step_id)
        if result and result.contents:
            return result.contents[0]
        return None

    # Extra context access
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from extra context."""
        return self.extra.get(key, default)

    # Messaging methods
    async def send_progress(self, progress: float, message: str = "") -> None:
        """Send progress update to user."""
        if not self._emitter:
            return

        from agentic_chatbot.events.models import ToolProgressEvent

        event = ToolProgressEvent.create(
            tool=self._operator_name,
            progress=progress,
            message=message,
            request_id=self._request_id,
        )
        await self._emitter.emit(event)

    async def send_content(
        self,
        content: ContentBlock,
        direct_to_user: bool = False,
    ) -> None:
        """
        Send content to user.

        Args:
            content: Content to send
            direct_to_user: If True, bypasses writer (for widgets, images)
        """
        if not self._emitter:
            return

        from agentic_chatbot.events.models import ToolContentEvent

        if direct_to_user:
            self._has_sent_direct = True
            self._direct_responses.append(content)

        event = ToolContentEvent.create(
            tool=self._operator_name,
            content_type=content.content_type.value,
            data=content.data,
            encoding=content.encoding,
            metadata={**content.metadata, "direct_to_user": direct_to_user},
            request_id=self._request_id,
        )
        await self._emitter.emit(event)

    async def elicit(
        self,
        prompt: str,
        input_type: str = "text",
        options: list[str] | None = None,
        default: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> Any:
        """
        Request input from user.

        Args:
            prompt: Question to ask
            input_type: Expected type (text, choice, confirm)
            options: Options for choice
            default: Default value
            timeout_seconds: Timeout

        Returns:
            User's response, or None if cancelled/timeout
        """
        if not self._elicitation_manager:
            raise RuntimeError(
                f"Elicitation not supported for '{self._operator_name}'"
            )

        import asyncio
        import uuid

        from agentic_chatbot.mcp.models import ElicitationRequest

        request = ElicitationRequest(
            request_id=str(uuid.uuid4()),
            prompt=prompt,
            input_type=input_type,
            options=options,
            default=default,
            timeout_seconds=timeout_seconds,
        )

        pending = await self._elicitation_manager.create_elicitation(
            server_id="operator",
            tool_name=self._operator_name,
            request=request,
        )

        if self._emitter:
            from agentic_chatbot.events.models import MCPElicitationRequestEvent

            event = MCPElicitationRequestEvent.create(
                server_id="operator",
                tool_name=self._operator_name,
                elicitation_id=pending.elicitation_id,
                prompt=prompt,
                input_type=input_type,
                options=options,
                default=default,
                timeout_seconds=timeout_seconds,
                request_id=self._request_id,
            )
            await self._emitter.emit(event)

        try:
            response = await self._elicitation_manager.wait_for_response(
                pending.elicitation_id,
                timeout=timeout_seconds,
            )
            return None if response.cancelled else response.value
        except asyncio.TimeoutError:
            await self._elicitation_manager.cancel_elicitation(pending.elicitation_id)
            return None

    async def confirm(self, prompt: str, timeout_seconds: float = 60.0) -> bool:
        """Request confirmation from user."""
        result = await self.elicit(prompt, "confirm", timeout_seconds=timeout_seconds)
        return bool(result)


@dataclass
class TaskInfo:
    """
    Task delegation information from supervisor.

    Consolidates TaskContext fields into a simpler structure.
    """

    description: str  # What to do
    goal: str  # Expected outcome
    scope: str = ""  # What's in/out of scope
    constraints: list[str] = field(default_factory=list)
    original_query: str = ""  # Reference to user's original request

    def to_prompt(self) -> str:
        """Format as prompt section."""
        parts = [f"## Task\n{self.description}", f"\n## Goal\n{self.goal}"]
        if self.scope:
            parts.append(f"\n## Scope\n{self.scope}")
        if self.constraints:
            constraints_text = "\n".join(f"- {c}" for c in self.constraints)
            parts.append(f"\n## Constraints\n{constraints_text}")
        if self.original_query:
            parts.append(f"\n## Context\nUser's request: {self.original_query[:200]}")
        return "\n".join(parts)


@dataclass
class ExecutionOutput:
    """
    Output from operator/tool execution.

    Consolidates:
    - OperatorResult: output, contents, success, error
    - ToolResult: contents, error, duration_ms
    - Token tracking

    Usage:
        # Success with text
        return ExecutionOutput.success(TextContent.markdown("# Result"))

        # Success with multiple contents
        return ExecutionOutput.success_multi([
            TextContent.markdown("# Analysis"),
            ImageContent.png(chart_data, "Chart"),
        ])

        # Error
        return ExecutionOutput.error("Tool execution failed")

        # With sourced content (for citations)
        return ExecutionOutput.success_sourced(
            content="Search results...",
            source_type="web_search",
            query_used="python async tutorial",
        )
    """

    status: ExecutionStatus = ExecutionStatus.SUCCESS
    contents: list[ContentBlock] = field(default_factory=list)

    # Sourced contents (with provenance for citations)
    sourced_contents: list[SourcedContent] = field(default_factory=list)

    # Error info
    error: str | None = None
    error_type: str | None = None

    # Direct responses (sent during execution, bypassing writer)
    direct_responses: list[ContentBlock] = field(default_factory=list)
    sent_direct_response: bool = False

    # Metrics
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Properties
    @property
    def success(self) -> bool:
        """Check if execution succeeded."""
        return self.status == ExecutionStatus.SUCCESS

    @property
    def primary_content(self) -> ContentBlock | None:
        """Get the primary content block."""
        if self.contents:
            return self.contents[0]
        if self.sourced_contents:
            return self.sourced_contents[0].content
        return None

    @property
    def text(self) -> str:
        """Get combined text from all contents."""
        texts = []
        for c in self.contents:
            if c.is_text:
                texts.append(c.as_text)
        for sc in self.sourced_contents:
            if sc.content.is_text:
                texts.append(sc.text)
        return "\n\n".join(texts)

    @property
    def should_skip_writer(self) -> bool:
        """Check if writer should be skipped."""
        return self.sent_direct_response

    # Factory methods
    @classmethod
    def success(
        cls,
        content: ContentBlock | str,
        **kwargs: Any,
    ) -> "ExecutionOutput":
        """Create success output with single content."""
        if isinstance(content, str):
            content = TextContent.markdown(content)
        return cls(
            status=ExecutionStatus.SUCCESS,
            contents=[content],
            **kwargs,
        )

    @classmethod
    def success_multi(
        cls,
        contents: list[ContentBlock],
        **kwargs: Any,
    ) -> "ExecutionOutput":
        """Create success output with multiple contents."""
        return cls(
            status=ExecutionStatus.SUCCESS,
            contents=contents,
            **kwargs,
        )

    @classmethod
    def success_sourced(
        cls,
        content: str | ContentBlock,
        source_type: str,
        source_id: str | None = None,
        query_used: str = "",
        **kwargs: Any,
    ) -> "ExecutionOutput":
        """Create success output with sourced content (for citations)."""
        from agentic_chatbot.data.sourced import create_sourced_content

        sourced = create_sourced_content(
            content=content,
            source_type=source_type,
            source_id=source_id,
            query_used=query_used,
        )
        return cls(
            status=ExecutionStatus.SUCCESS,
            sourced_contents=[sourced],
            **kwargs,
        )

    @classmethod
    def error(
        cls,
        message: str,
        error_type: str = "execution",
        **kwargs: Any,
    ) -> "ExecutionOutput":
        """Create error output."""
        return cls(
            status=ExecutionStatus.ERROR,
            contents=[ErrorContent(message, error_type)],
            error=message,
            error_type=error_type,
            **kwargs,
        )

    @classmethod
    def timeout(cls, message: str = "Operation timed out") -> "ExecutionOutput":
        """Create timeout output."""
        return cls(
            status=ExecutionStatus.TIMEOUT,
            contents=[ErrorContent.timeout(message)],
            error=message,
            error_type="timeout",
        )

    @classmethod
    def blocked(cls, reason: str = "Waiting for user input") -> "ExecutionOutput":
        """Create blocked output."""
        return cls(
            status=ExecutionStatus.BLOCKED,
            error=reason,
            error_type="blocked",
        )

    @classmethod
    def with_direct_response(
        cls,
        content: ContentBlock | str,
        direct_responses: list[ContentBlock],
        **kwargs: Any,
    ) -> "ExecutionOutput":
        """Create output that sent direct response to user."""
        if isinstance(content, str):
            content = TextContent.markdown(content)
        return cls(
            status=ExecutionStatus.SUCCESS,
            contents=[content],
            direct_responses=direct_responses,
            sent_direct_response=True,
            **kwargs,
        )
