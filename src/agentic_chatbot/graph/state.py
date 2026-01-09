"""LangGraph state definitions for the agentic chatbot.

This module defines the shared state that flows through the graph.
Uses TypedDict with Annotated reducers for proper state management.

Key Concepts:
- State is a shared memory object that flows through all nodes
- Reducers (operator.add) define how new data combines with existing data
- For lists: append behavior; for strings: concatenation

Data Model:
- ContentBlock: Atomic unit of information (from data.content)
- SourcedContent: ContentBlock with provenance (from data.sourced)
- Directive: Supervisor decision (from data.directive)
- ExecutionOutput: Operator/tool result (from data.execution)
"""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Any, Annotated, Literal, Sequence
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, AnyMessage

from agentic_chatbot.events.emitter import EventEmitter
from agentic_chatbot.mcp.callbacks import MCPCallbacks, ElicitationManager
from agentic_chatbot.core.workflow import WorkflowDefinition, WorkflowResult
from agentic_chatbot.config.models import TokenUsage

# New unified data model
from agentic_chatbot.data.content import ContentBlock
from agentic_chatbot.data.sourced import SourcedContent, ContentSummary
from agentic_chatbot.data.execution import ExecutionOutput, TaskInfo
from agentic_chatbot.data.directive import Directive, DirectiveType, DirectiveRecord

# Document types (optional import for type hints)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agentic_chatbot.documents.models import DocumentSummary, LoadedDocument
    from agentic_chatbot.cognition.models import CognitiveContext
    from agentic_chatbot.cognition.service import CognitionService


# =============================================================================
# DECISION AND RESULT MODELS
# =============================================================================


class SupervisorDecision(BaseModel):
    """
    Schema for Supervisor's action decision.

    DEPRECATED: Use Directive from agentic_chatbot.data.directive instead.
    Kept for backward compatibility during migration.
    """

    action: Literal["ANSWER", "CALL_TOOL", "CREATE_WORKFLOW", "CLARIFY"]
    reasoning: str = Field(..., description="Explanation of the decision")

    # For ANSWER
    response: str | None = Field(None, description="Direct response for ANSWER action")

    # For CALL_TOOL - includes task context for the operator
    operator: str | None = Field(None, description="Operator name for CALL_TOOL action")
    params: dict[str, Any] | None = Field(None, description="Parameters for the operator")

    # Task delegation context (for CALL_TOOL and CREATE_WORKFLOW)
    task_description: str | None = Field(
        None, description="Reformulated task description for the operator"
    )
    task_goal: str | None = Field(None, description="Expected outcome from this action")
    task_scope: str | None = Field(None, description="What's in/out of scope")

    # For CREATE_WORKFLOW
    goal: str | None = Field(None, description="Workflow goal for CREATE_WORKFLOW action")
    steps: list[dict[str, Any]] | None = Field(
        None, description="Workflow steps for CREATE_WORKFLOW action"
    )

    # For CLARIFY
    question: str | None = Field(None, description="Clarification question for CLARIFY action")

    def to_task_info(self, original_query: str = "") -> TaskInfo:
        """Convert decision to TaskInfo for operator."""
        return TaskInfo(
            description=self.task_description or self.reasoning,
            goal=self.task_goal or "Complete the requested operation",
            scope=self.task_scope or "",
            original_query=original_query[:200] if original_query else "",
        )

    def to_directive(self) -> Directive:
        """Convert to new Directive type."""
        action_map = {
            "ANSWER": DirectiveType.ANSWER,
            "CALL_TOOL": DirectiveType.CALL_OPERATOR,
            "CREATE_WORKFLOW": DirectiveType.CREATE_WORKFLOW,
            "CLARIFY": DirectiveType.CLARIFY,
        }
        return Directive(
            directive_type=action_map[self.action],
            reasoning=self.reasoning,
            response=self.response,
            operator=self.operator,
            params=self.params or {},
            task=self.to_task_info() if self.operator else None,
            goal=self.goal,
            steps=self.steps or [],
            question=self.question,
        )


class ToolResult(BaseModel):
    """
    Result from a tool execution.

    DEPRECATED: Use ExecutionOutput from agentic_chatbot.data.execution instead.
    Kept for backward compatibility during migration.
    """

    tool_name: str
    success: bool
    content: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WorkflowStep(BaseModel):
    """A step in a multi-step workflow."""

    step_id: str
    name: str
    operator: str
    params: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    result: ToolResult | None = None


class ReflectionResult(BaseModel):
    """Result from the reflection node."""

    assessment: Literal["satisfied", "need_more", "blocked"]
    reasoning: str
    suggested_action: str | None = None
    missing_info: list[str] = Field(default_factory=list)


class ActionRecord(BaseModel):
    """Record of an action taken during the conversation."""

    action: str
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    iteration: int
    result: dict[str, Any] | None = None


# =============================================================================
# CUSTOM REDUCERS
# =============================================================================


def reduce_messages(
    left: Sequence[BaseMessage] | None,
    right: Sequence[BaseMessage] | None,
) -> list[BaseMessage]:
    """Reducer for message lists - appends new messages."""
    if not left:
        left = []
    if not right:
        right = []
    return list(left) + list(right)


def reduce_tool_results(
    left: list[ToolResult] | None,
    right: list[ToolResult] | None,
) -> list[ToolResult]:
    """Reducer for tool results - appends new results."""
    if not left:
        left = []
    if not right:
        right = []
    return left + right


def reduce_action_history(
    left: list[ActionRecord] | None,
    right: list[ActionRecord] | None,
) -> list[ActionRecord]:
    """Reducer for action history - appends new records."""
    if not left:
        left = []
    if not right:
        right = []
    return left + right


def reduce_workflow_steps(
    left: list[WorkflowStep] | None,
    right: list[WorkflowStep] | None,
) -> list[WorkflowStep]:
    """Reducer for workflow steps - merges by step_id."""
    if not left:
        left = []
    if not right:
        return list(left)

    # Create lookup by step_id
    steps_by_id = {s.step_id: s for s in left}

    # Update or add from right
    for step in right:
        steps_by_id[step.step_id] = step

    return list(steps_by_id.values())


def reduce_sourced_contents(
    left: list[SourcedContent] | None,
    right: list[SourcedContent] | None,
) -> list[SourcedContent]:
    """Reducer for sourced contents - appends new items."""
    if not left:
        left = []
    if not right:
        right = []
    return left + right


def reduce_execution_outputs(
    left: list[ExecutionOutput] | None,
    right: list[ExecutionOutput] | None,
) -> list[ExecutionOutput]:
    """Reducer for execution outputs - appends new results."""
    if not left:
        left = []
    if not right:
        right = []
    return left + right


def reduce_directive_records(
    left: list[DirectiveRecord] | None,
    right: list[DirectiveRecord] | None,
) -> list[DirectiveRecord]:
    """Reducer for directive records - appends new records."""
    if not left:
        left = []
    if not right:
        right = []
    return left + right


def reduce_token_usage(
    left: TokenUsage | None,
    right: TokenUsage | None,
) -> TokenUsage:
    """Reducer for token usage - accumulates all token counts."""
    if not left:
        left = TokenUsage()
    if not right:
        return left
    return left + right


def reduce_document_summaries(
    left: list[Any] | None,
    right: list[Any] | None,
) -> list[Any]:
    """
    Reducer for document summaries.

    Merges by document_id to avoid duplicates.
    Right values override left for the same document_id.
    """
    if not left:
        left = []
    if not right:
        return list(left)

    # Create lookup by document_id
    by_id = {s.document_id: s for s in left}

    # Update or add from right
    for summary in right:
        by_id[summary.document_id] = summary

    return list(by_id.values())


def reduce_loaded_documents(
    left: list[Any] | None,
    right: list[Any] | None,
) -> list[Any]:
    """
    Reducer for loaded documents.

    Merges by document_id. Right values override left for the same document_id.
    """
    if not left:
        left = []
    if not right:
        return list(left)

    # Create lookup by document_id
    by_id = {d.document_id: d for d in left}

    # Update or add from right
    for doc in right:
        by_id[doc.document_id] = doc

    return list(by_id.values())


# =============================================================================
# MAIN STATE DEFINITION
# =============================================================================


class ChatState(TypedDict, total=False):
    """
    Main state that flows through the LangGraph.

    The state is divided into sections:
    - Input: Initial user request data
    - Conversation: Message history with reducer for appending
    - Supervisor: Decision-making state
    - Execution: Tool and workflow execution state
    - Output: Final response data
    - Context: Runtime context (MCP, events, etc.)

    Fields with Annotated[..., operator.add] will append rather than replace.
    """

    # -------------------------------------------------------------------------
    # INPUT
    # -------------------------------------------------------------------------
    user_query: str  # The user's message
    conversation_id: str  # Unique conversation identifier
    request_id: str  # Unique request identifier
    user_id: str | None  # User identifier for personalization
    user_context: dict[str, Any]  # Additional context from the user
    requested_model: str | None  # Model requested by user for response generation

    # -------------------------------------------------------------------------
    # QUERY UNDERSTANDING
    # Deep analysis of user query before action
    # -------------------------------------------------------------------------
    query_understanding: Any | None  # QueryUnderstanding object
    needs_clarification: bool  # Whether to ask clarification before proceeding
    clarification_questions: list[str]  # Questions to ask user

    # -------------------------------------------------------------------------
    # CONVERSATION HISTORY
    # Using reducer to append messages rather than replace
    # -------------------------------------------------------------------------
    messages: Annotated[list[BaseMessage], reduce_messages]

    # -------------------------------------------------------------------------
    # SUPERVISOR STATE
    # -------------------------------------------------------------------------
    current_decision: SupervisorDecision | None  # Current supervisor decision
    iteration: int  # Current iteration count (for loop control)
    max_iterations: int  # Maximum allowed iterations
    action_history: Annotated[list[ActionRecord], reduce_action_history]

    # -------------------------------------------------------------------------
    # EXECUTION STATE
    # -------------------------------------------------------------------------
    tool_results: Annotated[list[ToolResult], reduce_tool_results]
    current_tool: str | None  # Currently executing tool
    current_params: dict[str, Any] | None  # Parameters for current tool

    # Workflow execution
    workflow_goal: str | None
    workflow_steps: Annotated[list[WorkflowStep], reduce_workflow_steps]
    workflow_completed: bool
    workflow_definition: WorkflowDefinition | None  # Planned workflow
    workflow_result: WorkflowResult | None  # Execution result

    # Reflection
    reflection: ReflectionResult | None

    # -------------------------------------------------------------------------
    # UNIFIED DATA MODEL
    # Uses SourcedContent as the atomic unit with provenance tracking
    # -------------------------------------------------------------------------

    # Current task delegation info (from supervisor's directive)
    current_task: TaskInfo | None

    # Sourced contents with provenance tracking (replaces data_chunks + data_summaries)
    # Each SourcedContent contains: content (ContentBlock) + source + optional summary
    sourced_contents: Annotated[list[SourcedContent], reduce_sourced_contents]

    # Execution outputs from operators/tools (replaces tool_results for new code)
    execution_outputs: Annotated[list[ExecutionOutput], reduce_execution_outputs]

    # Directive history (replaces action_history for new code)
    directive_history: Annotated[list[DirectiveRecord], reduce_directive_records]

    # Source counter for generating unique citation IDs
    source_counter: dict[str, int]

    # Current directive (replaces current_decision for new code)
    current_directive: Directive | None

    # -------------------------------------------------------------------------
    # DOCUMENT CONTEXT
    # Documents uploaded by user for conversation context
    # -------------------------------------------------------------------------

    # Document summaries for supervisor decision-making
    # These are lightweight summaries used to decide which documents to load
    document_summaries: Annotated[list[Any], reduce_document_summaries]

    # Loaded document content for context
    # These are full or partial documents loaded based on supervisor decisions
    loaded_documents: Annotated[list[Any], reduce_loaded_documents]

    # -------------------------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------------------------
    final_response: str  # Final response to user
    clarify_question: str | None  # Question for clarification
    response_chunks: Annotated[list[str], operator.add]  # For streaming

    # Direct response tracking (operator/tool bypassed writer)
    sent_direct_response: bool  # Whether operator sent response directly to user
    direct_response_contents: list[Any]  # Content items sent directly

    # -------------------------------------------------------------------------
    # TOKEN TRACKING
    # Accumulates all token usage across the entire request
    # -------------------------------------------------------------------------
    token_usage: Annotated[TokenUsage, reduce_token_usage]

    # -------------------------------------------------------------------------
    # RUNTIME CONTEXT (not persisted to checkpointer)
    # -------------------------------------------------------------------------
    event_emitter: EventEmitter | None  # For SSE events
    event_queue: Any | None  # asyncio.Queue for events
    mcp_registry: Any | None  # MCPServerRegistry
    mcp_session_manager: Any | None  # MCPSessionManager
    mcp_callbacks: MCPCallbacks | None  # MCP callback handlers
    elicitation_manager: ElicitationManager | None  # For user input requests
    tool_provider: Any | None  # UnifiedToolProvider for local + remote tools

    # Error handling
    error: str | None
    error_type: str | None


# =============================================================================
# STATE INITIALIZATION HELPERS
# =============================================================================


def create_initial_state(
    user_query: str,
    conversation_id: str,
    request_id: str,
    event_emitter: EventEmitter | None = None,
    event_queue: Any = None,
    mcp_registry: Any = None,
    mcp_session_manager: Any = None,
    mcp_callbacks: MCPCallbacks | None = None,
    elicitation_manager: ElicitationManager | None = None,
    tool_provider: Any = None,
    user_context: dict[str, Any] | None = None,
    requested_model: str | None = None,
    user_id: str | None = None,
) -> ChatState:
    """
    Create initial state for a new chat request.

    Args:
        user_query: The user's message
        conversation_id: Unique conversation identifier
        request_id: Unique request identifier
        event_emitter: EventEmitter for SSE streaming
        event_queue: Queue for events
        mcp_registry: MCP server registry
        mcp_session_manager: MCP session manager
        mcp_callbacks: MCP callbacks
        tool_provider: UnifiedToolProvider for local + remote tools
        elicitation_manager: Manager for user input requests
        user_context: Additional context from user
        requested_model: Model ID requested by user for response generation
        user_id: User identifier for personalization

    Returns:
        Initialized ChatState
    """
    return ChatState(
        # Input
        user_query=user_query,
        conversation_id=conversation_id,
        request_id=request_id,
        user_id=user_id,
        user_context=user_context or {},
        requested_model=requested_model,

        # Query understanding
        query_understanding=None,
        needs_clarification=False,
        clarification_questions=[],

        # Conversation
        messages=[],

        # Supervisor
        current_decision=None,
        iteration=0,
        max_iterations=5,
        action_history=[],

        # Execution
        tool_results=[],
        current_tool=None,
        current_params=None,
        workflow_goal=None,
        workflow_steps=[],
        workflow_completed=False,
        workflow_definition=None,
        workflow_result=None,
        reflection=None,

        # Unified data model
        current_task=None,
        sourced_contents=[],
        execution_outputs=[],
        directive_history=[],
        source_counter={},
        current_directive=None,

        # Document context
        document_summaries=[],
        loaded_documents=[],

        # Output
        final_response="",
        clarify_question=None,
        response_chunks=[],

        # Direct response tracking
        sent_direct_response=False,
        direct_response_contents=[],

        # Token tracking
        token_usage=TokenUsage(),

        # Runtime context
        event_emitter=event_emitter,
        event_queue=event_queue,
        mcp_registry=mcp_registry,
        mcp_session_manager=mcp_session_manager,
        mcp_callbacks=mcp_callbacks,
        elicitation_manager=elicitation_manager,
        tool_provider=tool_provider,

        # Error handling
        error=None,
        error_type=None,
    )


def get_emitter(state: ChatState) -> EventEmitter | None:
    """Get EventEmitter from state, creating one if needed."""
    emitter = state.get("event_emitter")
    if emitter:
        return emitter

    queue = state.get("event_queue")
    if queue:
        return EventEmitter(queue)

    return None


def generate_source_id(state: ChatState, source_type: str) -> tuple[str, dict[str, int]]:
    """
    Generate unique source ID for citations.

    Args:
        state: Current chat state
        source_type: Type of source (e.g., "web_search", "rag")

    Returns:
        Tuple of (source_id, updated_counter)
    """
    counter = state.get("source_counter", {}).copy()
    count = counter.get(source_type, 0) + 1
    counter[source_type] = count
    source_id = f"{source_type}_{count}"
    return source_id, counter


def get_summaries_text(state: ChatState) -> str:
    """Get formatted summaries for supervisor context."""
    sourced = state.get("sourced_contents", [])
    if not sourced:
        return "No data collected yet."

    texts = []
    for sc in sourced:
        texts.append(sc.to_supervisor_text())
    return "\n\n".join(texts)


def get_sourced_contents(state: ChatState) -> list[SourcedContent]:
    """Get all sourced contents for synthesizer/writer."""
    return state.get("sourced_contents", [])


def get_execution_outputs(state: ChatState) -> list[ExecutionOutput]:
    """Get all execution outputs."""
    return state.get("execution_outputs", [])


def get_citation_blocks(state: ChatState) -> str:
    """Get all citation blocks for footnotes."""
    sourced = state.get("sourced_contents", [])
    return "\n\n".join(sc.citation_block for sc in sourced)


def get_document_summaries_text(state: ChatState) -> str:
    """
    Get formatted document summaries for supervisor context.

    Returns a text representation of all document summaries that helps
    the supervisor decide which documents to load.
    """
    summaries = state.get("document_summaries", [])
    if not summaries:
        return "No documents uploaded for this conversation."

    lines = [f"## Uploaded Documents ({len(summaries)} total)\n"]

    for summary in summaries:
        lines.append(f"### [{summary.document_id}] {summary.filename}")
        lines.append(f"**Status:** {summary.status.value}")

        if summary.status.value == "ready":
            lines.append(f"**Type:** {summary.document_type}")
            lines.append(f"**Summary:** {summary.overall_summary}")
            lines.append(f"**Topics:** {', '.join(summary.key_topics)}")
            lines.append(f"**Size:** {summary.chunk_count} chunks, ~{summary.total_tokens} tokens")
            if summary.relevance_hints:
                lines.append(f"**When to use:** {summary.relevance_hints}")
        else:
            lines.append(f"**Progress:** {summary.processing_progress:.0%}")

        lines.append("")  # Blank line between documents

    return "\n".join(lines)


def get_loaded_documents_text(state: ChatState) -> str:
    """
    Get formatted loaded document content for context assembly.

    Returns the full content of all loaded documents.
    """
    documents = state.get("loaded_documents", [])
    if not documents:
        return ""

    lines = ["## Document Content\n"]

    for doc in documents:
        lines.append(f"### {doc.filename} (ID: {doc.document_id})")
        if not doc.is_complete:
            lines.append(f"*Partial content: chunks {doc.chunks_loaded}*")
        lines.append("")
        lines.append(doc.content)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def get_document_context(state: ChatState) -> str:
    """
    Get combined document context for supervisor/synthesizer.

    Returns both summaries and loaded content formatted for context.
    """
    summaries_text = get_document_summaries_text(state)
    loaded_text = get_loaded_documents_text(state)

    if not loaded_text:
        return summaries_text

    return f"{summaries_text}\n\n{loaded_text}"
