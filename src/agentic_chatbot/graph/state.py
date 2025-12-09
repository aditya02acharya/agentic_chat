"""LangGraph state definitions for the agentic chatbot.

This module defines the shared state that flows through the graph.
Uses TypedDict with Annotated reducers for proper state management.

Key Concepts:
- State is a shared memory object that flows through all nodes
- Reducers (operator.add) define how new data combines with existing data
- For lists: append behavior; for strings: concatenation
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


# =============================================================================
# DECISION AND RESULT MODELS
# =============================================================================


class SupervisorDecision(BaseModel):
    """Schema for Supervisor's action decision."""

    action: Literal["ANSWER", "CALL_TOOL", "CREATE_WORKFLOW", "CLARIFY"]
    reasoning: str = Field(..., description="Explanation of the decision")

    # For ANSWER
    response: str | None = Field(None, description="Direct response for ANSWER action")

    # For CALL_TOOL
    operator: str | None = Field(None, description="Operator name for CALL_TOOL action")
    params: dict[str, Any] | None = Field(None, description="Parameters for the operator")

    # For CREATE_WORKFLOW
    goal: str | None = Field(None, description="Workflow goal for CREATE_WORKFLOW action")
    steps: list[dict[str, Any]] | None = Field(
        None, description="Workflow steps for CREATE_WORKFLOW action"
    )

    # For CLARIFY
    question: str | None = Field(None, description="Clarification question for CLARIFY action")


class ToolResult(BaseModel):
    """Result from a tool execution."""

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
    user_context: dict[str, Any]  # Additional context from the user

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

    # Reflection
    reflection: ReflectionResult | None

    # -------------------------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------------------------
    final_response: str  # Final response to user
    clarify_question: str | None  # Question for clarification
    response_chunks: Annotated[list[str], operator.add]  # For streaming

    # -------------------------------------------------------------------------
    # RUNTIME CONTEXT (not persisted to checkpointer)
    # -------------------------------------------------------------------------
    event_emitter: EventEmitter | None  # For SSE events
    event_queue: Any | None  # asyncio.Queue for events
    mcp_registry: Any | None  # MCPServerRegistry
    mcp_session_manager: Any | None  # MCPSessionManager
    mcp_callbacks: MCPCallbacks | None  # MCP callback handlers
    elicitation_manager: ElicitationManager | None  # For user input requests

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
    user_context: dict[str, Any] | None = None,
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
        elicitation_manager: Manager for user input requests
        user_context: Additional context from user

    Returns:
        Initialized ChatState
    """
    return ChatState(
        # Input
        user_query=user_query,
        conversation_id=conversation_id,
        request_id=request_id,
        user_context=user_context or {},

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
        reflection=None,

        # Output
        final_response="",
        clarify_question=None,
        response_chunks=[],

        # Runtime context
        event_emitter=event_emitter,
        event_queue=event_queue,
        mcp_registry=mcp_registry,
        mcp_session_manager=mcp_session_manager,
        mcp_callbacks=mcp_callbacks,
        elicitation_manager=elicitation_manager,

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
