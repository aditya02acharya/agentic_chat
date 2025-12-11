"""Operator context and result models."""

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field

from agentic_chatbot.mcp.models import ToolContent, OutputDataType, ElicitationRequest

if TYPE_CHECKING:
    from agentic_chatbot.events.emitter import EventEmitter
    from agentic_chatbot.mcp.callbacks import ElicitationManager


# =============================================================================
# MESSAGING CONTEXT
# =============================================================================


class MessagingContext:
    """
    Context for operators and tools to send messages directly to users.

    This allows operators/tools to:
    - Send direct responses (bypassing the writer)
    - Report progress updates
    - Request user input (elicitation)
    - Stream content incrementally
    - Send rich content (images, widgets, HTML)

    Usage in an operator:
        async def execute(self, context: OperatorContext, ...) -> OperatorResult:
            # Send progress update
            await context.messaging.send_progress(0.5, "Processing data...")

            # Send a widget directly to user
            await context.messaging.send_content(
                widget_data,
                content_type="application/vnd.mcp.widget+json",
                direct_to_user=True
            )

            # Request user input
            response = await context.messaging.elicit(
                prompt="Please confirm this action",
                input_type="confirm"
            )
    """

    def __init__(
        self,
        emitter: "EventEmitter | None" = None,
        elicitation_manager: "ElicitationManager | None" = None,
        request_id: str | None = None,
        operator_name: str = "unknown",
    ):
        """
        Initialize messaging context.

        Args:
            emitter: EventEmitter for sending events to SSE stream
            elicitation_manager: Manager for user input requests
            request_id: Request ID for event correlation
            operator_name: Name of the operator/tool using this context
        """
        self._emitter = emitter
        self._elicitation_manager = elicitation_manager
        self._request_id = request_id
        self._operator_name = operator_name
        self._direct_responses: list[ToolContent] = []
        self._has_sent_direct_response = False

    @property
    def is_enabled(self) -> bool:
        """Check if messaging is enabled (emitter is available)."""
        return self._emitter is not None

    @property
    def has_direct_responses(self) -> bool:
        """Check if any direct responses have been sent."""
        return self._has_sent_direct_response

    @property
    def direct_responses(self) -> list[ToolContent]:
        """Get list of direct responses that were sent."""
        return self._direct_responses.copy()

    async def send_progress(self, progress: float, message: str = "") -> None:
        """
        Send a progress update to the user.

        Args:
            progress: Progress value between 0.0 and 1.0
            message: Optional progress message
        """
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
        data: Any,
        content_type: str = "text/plain",
        encoding: str | None = None,
        metadata: dict[str, Any] | None = None,
        direct_to_user: bool = False,
    ) -> None:
        """
        Send content to the user.

        Args:
            data: Content data
            content_type: MIME type of the content
            encoding: Encoding type (e.g., "base64" for binary)
            metadata: Additional metadata
            direct_to_user: If True, content is sent directly (bypasses writer)
        """
        if not self._emitter:
            return

        from agentic_chatbot.events.models import ToolContentEvent

        # Track direct responses
        if direct_to_user:
            self._has_sent_direct_response = True
            content = ToolContent(
                content_type=content_type,
                data=data,
                encoding=encoding,
                metadata=metadata or {},
            )
            self._direct_responses.append(content)

        event = ToolContentEvent.create(
            tool=self._operator_name,
            content_type=content_type,
            data=data,
            encoding=encoding,
            metadata={**(metadata or {}), "direct_to_user": direct_to_user},
            request_id=self._request_id,
        )
        await self._emitter.emit(event)

    async def send_text(self, text: str, direct_to_user: bool = False) -> None:
        """
        Send text content to the user.

        Args:
            text: Text to send
            direct_to_user: If True, text is sent directly (bypasses writer)
        """
        await self.send_content(
            data=text,
            content_type="text/plain",
            direct_to_user=direct_to_user,
        )

    async def send_markdown(self, markdown: str, direct_to_user: bool = False) -> None:
        """
        Send markdown content to the user.

        Args:
            markdown: Markdown text to send
            direct_to_user: If True, content is sent directly (bypasses writer)
        """
        await self.send_content(
            data=markdown,
            content_type="text/markdown",
            direct_to_user=direct_to_user,
        )

    async def send_html(self, html: str, direct_to_user: bool = False) -> None:
        """
        Send HTML content to the user.

        Args:
            html: HTML content to send
            direct_to_user: If True, content is sent directly (bypasses writer)
        """
        await self.send_content(
            data=html,
            content_type="text/html",
            direct_to_user=direct_to_user,
        )

    async def send_image(
        self,
        base64_data: str,
        mime_type: str = "image/png",
        alt_text: str = "",
        direct_to_user: bool = True,
    ) -> None:
        """
        Send an image to the user.

        Args:
            base64_data: Base64-encoded image data
            mime_type: Image MIME type
            alt_text: Alternative text for the image
            direct_to_user: If True, image is sent directly (bypasses writer)
        """
        await self.send_content(
            data=base64_data,
            content_type=mime_type,
            encoding="base64",
            metadata={"alt_text": alt_text},
            direct_to_user=direct_to_user,
        )

    async def send_widget(
        self,
        widget_spec: dict[str, Any],
        direct_to_user: bool = True,
    ) -> None:
        """
        Send an interactive widget to the user.

        Args:
            widget_spec: Widget specification (type, data, actions)
            direct_to_user: If True, widget is sent directly (bypasses writer)
        """
        await self.send_content(
            data=widget_spec,
            content_type="application/vnd.mcp.widget+json",
            direct_to_user=direct_to_user,
        )

    async def elicit(
        self,
        prompt: str,
        input_type: str = "text",
        options: list[str] | None = None,
        default: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> Any:
        """
        Request input from the user and wait for response.

        Args:
            prompt: Question to ask the user
            input_type: Expected input type (text, choice, confirm)
            options: Options for choice input
            default: Default value
            timeout_seconds: Timeout for response

        Returns:
            User's response value, or None if cancelled/timed out

        Raises:
            RuntimeError: If elicitation is not supported (no manager)
        """
        if not self._elicitation_manager:
            raise RuntimeError(
                f"Elicitation not supported for operator '{self._operator_name}'. "
                "ElicitationManager not provided."
            )

        import uuid

        request = ElicitationRequest(
            request_id=str(uuid.uuid4()),
            prompt=prompt,
            input_type=input_type,
            options=options,
            default=default,
            timeout_seconds=timeout_seconds,
        )

        # Create pending elicitation
        pending = await self._elicitation_manager.create_elicitation(
            server_id="operator",
            tool_name=self._operator_name,
            request=request,
        )

        # Emit event to UI
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

        # Wait for response
        import asyncio

        try:
            response = await self._elicitation_manager.wait_for_response(
                pending.elicitation_id,
                timeout=timeout_seconds,
            )
            if response.cancelled:
                return None
            return response.value
        except asyncio.TimeoutError:
            await self._elicitation_manager.cancel_elicitation(pending.elicitation_id)
            return None

    async def confirm(self, prompt: str, timeout_seconds: float = 60.0) -> bool:
        """
        Request confirmation from the user.

        Args:
            prompt: Confirmation question
            timeout_seconds: Timeout for response

        Returns:
            True if user confirmed, False otherwise
        """
        result = await self.elicit(
            prompt=prompt,
            input_type="confirm",
            timeout_seconds=timeout_seconds,
        )
        return bool(result)

    async def choose(
        self,
        prompt: str,
        options: list[str],
        default: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> str | None:
        """
        Request user to choose from options.

        Args:
            prompt: Question to ask
            options: List of options to choose from
            default: Default option
            timeout_seconds: Timeout for response

        Returns:
            Selected option, or None if cancelled/timed out
        """
        result = await self.elicit(
            prompt=prompt,
            input_type="choice",
            options=options,
            default=default,
            timeout_seconds=timeout_seconds,
        )
        return result


# =============================================================================
# OPERATOR CONTEXT
# =============================================================================


class OperatorContext(BaseModel):
    """
    Context provided to an operator during execution.

    Built by ContextAssembler based on operator's context_requirements.

    Includes MessagingContext for operators to:
    - Send direct responses to users (bypassing writer)
    - Report progress updates
    - Request user input (elicitation)
    - Stream content incrementally
    """

    model_config = {"arbitrary_types_allowed": True}

    # Core query
    query: str = Field(..., description="The user's query")

    # Conversation context
    recent_messages: list[dict[str, Any]] = Field(
        default_factory=list, description="Recent conversation messages"
    )
    conversation_summary: str = Field("", description="Summary of older conversation")

    # Tool context (lazy loaded)
    tool_schemas: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Full schemas for required tools"
    )

    # Results from previous steps (for workflows)
    step_results: dict[str, Any] = Field(
        default_factory=dict, description="Results from previous workflow steps"
    )

    # Additional context
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Additional context data"
    )

    # Shared store reference (for advanced use)
    shared_store: dict[str, Any] = Field(
        default_factory=dict, description="Reference to shared store"
    )

    # Messaging context for direct user communication
    # This is set by the system when the operator supports messaging capabilities
    _messaging: MessagingContext | None = None

    @property
    def messaging(self) -> MessagingContext:
        """
        Get messaging context for direct user communication.

        Returns a MessagingContext that allows the operator to:
        - Send direct responses (bypassing writer)
        - Report progress updates
        - Request user input (elicitation)
        - Stream content

        Returns:
            MessagingContext instance (may be a no-op if not configured)
        """
        if self._messaging is None:
            # Return a no-op messaging context if not configured
            self._messaging = MessagingContext()
        return self._messaging

    def set_messaging(self, messaging: MessagingContext) -> None:
        """
        Set the messaging context.

        Called by the system to wire up the messaging capabilities.

        Args:
            messaging: MessagingContext instance with emitter and manager
        """
        self._messaging = messaging

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from extra context."""
        return self.extra.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in extra context."""
        self.extra[key] = value


class OperatorResult(BaseModel):
    """
    Result from operator execution.

    Supports both simple outputs and multi-modal content.
    Includes flags for direct response handling.
    """

    # Main output (text or structured data)
    output: Any = Field(None, description="Main output from the operator")

    # Multi-modal contents (images, widgets, etc.)
    contents: list[ToolContent] = Field(
        default_factory=list, description="Multi-modal content items"
    )

    # Direct responses that were sent to user (bypassing writer)
    direct_responses: list[ToolContent] = Field(
        default_factory=list,
        description="Content items that were sent directly to user",
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Execution metadata"
    )

    # Status
    success: bool = Field(True, description="Whether execution succeeded")
    error: str | None = Field(None, description="Error message if failed")

    # Direct response flag - if True, the writer should be skipped
    sent_direct_response: bool = Field(
        False,
        description="Whether operator sent response directly to user (skip writer)",
    )

    # Token usage (for LLM operators)
    input_tokens: int = Field(0, description="Input tokens used")
    output_tokens: int = Field(0, description="Output tokens generated")

    @property
    def text_output(self) -> str:
        """Get output as text string."""
        if isinstance(self.output, str):
            return self.output
        if isinstance(self.output, dict):
            import json

            return json.dumps(self.output, indent=2)
        return str(self.output) if self.output else ""

    @property
    def has_contents(self) -> bool:
        """Check if result has multi-modal contents."""
        return len(self.contents) > 0

    @property
    def should_skip_writer(self) -> bool:
        """Check if the writer should be skipped for this result."""
        return self.sent_direct_response

    @classmethod
    def success_result(
        cls,
        output: Any,
        contents: list[ToolContent] | None = None,
        direct_responses: list[ToolContent] | None = None,
        sent_direct_response: bool = False,
        **kwargs: Any,
    ) -> "OperatorResult":
        """
        Create successful result.

        Args:
            output: Main output from the operator
            contents: Multi-modal content items
            direct_responses: Content items sent directly to user
            sent_direct_response: Whether to skip the writer
            **kwargs: Additional fields
        """
        return cls(
            output=output,
            contents=contents or [],
            direct_responses=direct_responses or [],
            sent_direct_response=sent_direct_response,
            success=True,
            **kwargs,
        )

    @classmethod
    def error_result(cls, error: str, **kwargs: Any) -> "OperatorResult":
        """Create error result."""
        return cls(
            success=False,
            error=error,
            **kwargs,
        )
