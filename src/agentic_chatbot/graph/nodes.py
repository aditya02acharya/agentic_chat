"""LangGraph node functions for the agentic chatbot.

Nodes are async functions that:
- Take the current state as input
- Return a partial state dict with updates
- Can emit events for SSE streaming

Each node focuses on a single responsibility following the original design.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from langchain_core.messages import HumanMessage, AIMessage

from agentic_chatbot.config.prompts import (
    SUPERVISOR_SYSTEM_PROMPT,
    SUPERVISOR_DECISION_PROMPT,
    REFLECT_SYSTEM_PROMPT,
    REFLECT_PROMPT,
    SYNTHESIZER_PROMPT,
    WRITER_PROMPT,
    BLOCKED_HANDLER_PROMPT,
)
from agentic_chatbot.events.models import (
    SupervisorThinkingEvent,
    SupervisorDecidedEvent,
    ToolStartEvent,
    ToolCompleteEvent,
    ToolErrorEvent,
    ResponseChunkEvent,
    ResponseDoneEvent,
    ClarifyRequestEvent,
)
from agentic_chatbot.graph.state import (
    ChatState,
    SupervisorDecision,
    ToolResult,
    ReflectionResult,
    ActionRecord,
    get_emitter,
)
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.operators.context import OperatorContext
from agentic_chatbot.utils.structured_llm import StructuredLLMCaller, StructuredOutputError
from agentic_chatbot.utils.llm import LLMClient
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def emit_event(state: ChatState, event: Any) -> None:
    """Emit an event to the SSE stream."""
    emitter = get_emitter(state)
    if emitter:
        await emitter.emit(event)


def format_conversation_context(state: ChatState) -> str:
    """Format conversation history for prompts."""
    messages = state.get("messages", [])
    if not messages:
        return "No previous conversation."

    formatted = []
    for msg in messages[-10:]:  # Last 10 messages
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = msg.content if hasattr(msg, "content") else str(msg)
        formatted.append(f"{role}: {content}")

    return "\n".join(formatted)


def format_tool_results(state: ChatState) -> str:
    """Format tool results for prompts."""
    results = state.get("tool_results", [])
    if not results:
        return "No tool results yet."

    formatted = []
    for result in results:
        status = "✓" if result.success else "✗"
        formatted.append(f"{status} {result.tool_name}: {result.content[:200]}...")

    return "\n".join(formatted)


def get_tool_summaries(state: ChatState) -> str:
    """Get formatted tool summaries."""
    registry = state.get("mcp_registry")
    if registry and hasattr(registry, "get_tool_summaries_text"):
        return registry.get_tool_summaries_text()
    return OperatorRegistry.get_operators_text()


# =============================================================================
# INITIALIZATION NODE
# =============================================================================


async def initialize_node(state: ChatState) -> dict[str, Any]:
    """
    Initialize the chat session.

    - Validates input
    - Sets up initial state values
    - Adds user message to conversation history
    """
    logger.info("Initializing chat session", request_id=state.get("request_id"))

    user_query = state.get("user_query", "")
    if not user_query.strip():
        return {
            "error": "Empty user query",
            "error_type": "validation",
        }

    # Add user message to conversation
    user_message = HumanMessage(content=user_query)

    return {
        "messages": [user_message],
        "iteration": 0,
    }


# =============================================================================
# SUPERVISOR NODE
# =============================================================================


async def supervisor_node(state: ChatState) -> dict[str, Any]:
    """
    Central decision-making agent using ReACT pattern.

    Analyzes the query and decides:
    - ANSWER: Direct response
    - CALL_TOOL: Use a single tool
    - CREATE_WORKFLOW: Multi-step execution
    - CLARIFY: Ask for clarification
    """
    await emit_event(
        state,
        SupervisorThinkingEvent.create(
            "Analyzing your question...",
            request_id=state.get("request_id"),
        ),
    )

    # Check iteration limit
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 5)

    if iteration >= max_iterations:
        logger.warning("Max iterations reached", iteration=iteration)
        return {
            "current_decision": SupervisorDecision(
                action="ANSWER",
                reasoning="Maximum iterations reached, providing best available answer",
                response="I've gathered the available information. Let me summarize what I found.",
            ),
            "iteration": iteration + 1,
        }

    # Build context
    tool_summaries = get_tool_summaries(state)
    conversation_context = format_conversation_context(state)
    tool_results = format_tool_results(state)

    # Format action history
    action_history = state.get("action_history", [])
    actions_text = (
        "\n".join(f"- {a.action}: {a.reasoning[:50]}..." for a in action_history)
        or "None"
    )

    # Build prompts
    system = SUPERVISOR_SYSTEM_PROMPT.format(tool_summaries=tool_summaries)
    prompt = SUPERVISOR_DECISION_PROMPT.format(
        query=state.get("user_query", ""),
        conversation_context=conversation_context,
        tool_results=tool_results,
        actions_this_turn=actions_text,
    )

    caller = StructuredLLMCaller(max_retries=3)

    try:
        decision = await caller.call(
            prompt=prompt,
            response_model=SupervisorDecision,
            system=system,
            model="sonnet",
        )

        # Emit decided event
        action_messages = {
            "ANSWER": "I can answer this directly",
            "CALL_TOOL": f"Using {decision.operator or 'tool'} to get information",
            "CREATE_WORKFLOW": "Creating a multi-step plan",
            "CLARIFY": "I need some clarification",
        }
        await emit_event(
            state,
            SupervisorDecidedEvent.create(
                action=decision.action,
                message=action_messages.get(decision.action, "Processing..."),
                request_id=state.get("request_id"),
            ),
        )

        logger.info("Supervisor decided", action=decision.action)

        # Record action
        action_record = ActionRecord(
            action=decision.action,
            reasoning=decision.reasoning,
            iteration=iteration,
        )

        return {
            "current_decision": decision,
            "current_tool": decision.operator,
            "current_params": decision.params,
            "workflow_goal": decision.goal,
            "iteration": iteration + 1,
            "action_history": [action_record],
        }

    except StructuredOutputError as e:
        logger.error(f"Supervisor decision failed: {e}")
        return {
            "current_decision": SupervisorDecision(
                action="CLARIFY",
                reasoning="Unable to determine appropriate action",
                question="Could you please rephrase your request?",
            ),
            "iteration": iteration + 1,
        }


# =============================================================================
# TOOL EXECUTION NODE
# =============================================================================


async def execute_tool_node(state: ChatState) -> dict[str, Any]:
    """
    Execute a single tool/operator.

    Uses the operator registry to find and execute the appropriate operator.
    """
    decision = state.get("current_decision")
    if not decision or not decision.operator:
        return {
            "tool_results": [
                ToolResult(
                    tool_name="unknown",
                    success=False,
                    error="No operator specified in decision",
                )
            ],
        }

    tool_name = decision.operator
    params = decision.params or {}

    await emit_event(
        state,
        ToolStartEvent.create(
            tool=tool_name,
            message=f"Executing {tool_name}...",
            request_id=state.get("request_id"),
        ),
    )

    try:
        # Get operator from registry
        operator_cls = OperatorRegistry.get(tool_name)
        if not operator_cls:
            raise ValueError(f"Operator {tool_name} not found")

        operator = operator_cls()

        # Create operator context
        context = OperatorContext(
            user_query=state.get("user_query", ""),
            params=params,
            conversation_id=state.get("conversation_id", ""),
            request_id=state.get("request_id", ""),
        )

        # Execute operator
        result = await operator.execute(context)

        await emit_event(
            state,
            ToolCompleteEvent.create(
                tool=tool_name,
                content_count=1,
                request_id=state.get("request_id"),
            ),
        )

        return {
            "tool_results": [
                ToolResult(
                    tool_name=tool_name,
                    success=result.success,
                    content=result.content,
                    error=result.error,
                    metadata=result.metadata,
                )
            ],
            "current_tool": None,
        }

    except Exception as e:
        logger.error(f"Tool execution failed: {e}", exc_info=True)

        await emit_event(
            state,
            ToolErrorEvent.create(
                tool=tool_name,
                error=str(e),
                request_id=state.get("request_id"),
            ),
        )

        return {
            "tool_results": [
                ToolResult(
                    tool_name=tool_name,
                    success=False,
                    error=str(e),
                )
            ],
            "current_tool": None,
        }


# =============================================================================
# REFLECTION NODE
# =============================================================================


async def reflect_node(state: ChatState) -> dict[str, Any]:
    """
    Analyze tool results and decide next action.

    Returns assessment:
    - satisfied: Have enough info to answer
    - need_more: Need additional tools/info
    - blocked: Cannot proceed
    """
    tool_results = state.get("tool_results", [])
    user_query = state.get("user_query", "")

    # Format results for reflection
    results_text = format_tool_results(state)

    system = REFLECT_SYSTEM_PROMPT
    prompt = REFLECT_PROMPT.format(
        query=user_query,
        tool_results=results_text,
        iteration=state.get("iteration", 0),
        max_iterations=state.get("max_iterations", 5),
    )

    caller = StructuredLLMCaller(max_retries=2)

    try:
        reflection = await caller.call(
            prompt=prompt,
            response_model=ReflectionResult,
            system=system,
            model="haiku",
        )

        logger.info("Reflection complete", assessment=reflection.assessment)

        return {"reflection": reflection}

    except StructuredOutputError:
        # Default to satisfied if reflection fails
        return {
            "reflection": ReflectionResult(
                assessment="satisfied",
                reasoning="Proceeding with available information",
            )
        }


# =============================================================================
# SYNTHESIS NODE
# =============================================================================


async def synthesize_node(state: ChatState) -> dict[str, Any]:
    """
    Synthesize tool results into a coherent response.

    Combines multiple tool outputs into a unified answer.
    """
    tool_results = state.get("tool_results", [])
    user_query = state.get("user_query", "")

    # Format results
    results_text = "\n\n".join(
        f"**{r.tool_name}**:\n{r.content}"
        for r in tool_results
        if r.success
    )

    prompt = SYNTHESIZER_PROMPT.format(
        query=user_query,
        tool_results=results_text,
    )

    client = LLMClient()
    response = await client.generate(prompt, model="sonnet")

    return {"final_response": response}


# =============================================================================
# WRITE NODE
# =============================================================================


async def write_node(state: ChatState) -> dict[str, Any]:
    """
    Format the final response for the user.

    Handles both direct answers and synthesized responses.
    """
    decision = state.get("current_decision")
    existing_response = state.get("final_response", "")

    # If we already have a synthesized response, use it
    if existing_response:
        final = existing_response
    elif decision and decision.action == "ANSWER" and decision.response:
        final = decision.response
    else:
        # Generate response based on context
        prompt = WRITER_PROMPT.format(
            query=state.get("user_query", ""),
            context=format_tool_results(state),
        )
        client = LLMClient()
        final = await client.generate(prompt, model="sonnet")

    # Add assistant message to conversation
    assistant_message = AIMessage(content=final)

    return {
        "final_response": final,
        "messages": [assistant_message],
    }


# =============================================================================
# STREAM NODE
# =============================================================================


async def stream_node(state: ChatState) -> dict[str, Any]:
    """
    Stream the final response to the user via SSE.

    Emits response chunks and completion event.
    """
    response = state.get("final_response", "")
    request_id = state.get("request_id")

    # Stream response in chunks
    chunk_size = 50
    for i in range(0, len(response), chunk_size):
        chunk = response[i : i + chunk_size]
        await emit_event(
            state,
            ResponseChunkEvent.create(content=chunk, request_id=request_id),
        )

    # Emit completion event
    await emit_event(
        state,
        ResponseDoneEvent.create(request_id=request_id),
    )

    return {"response_chunks": [response]}


# =============================================================================
# CLARIFY NODE
# =============================================================================


async def clarify_node(state: ChatState) -> dict[str, Any]:
    """
    Ask the user for clarification.

    Emits clarification request event.
    """
    decision = state.get("current_decision")
    question = decision.question if decision else "Could you please provide more details?"

    await emit_event(
        state,
        ClarifyRequestEvent.create(
            question=question,
            request_id=state.get("request_id"),
        ),
    )

    return {
        "clarify_question": question,
        "final_response": question,
    }


# =============================================================================
# BLOCKED HANDLER NODE
# =============================================================================


async def handle_blocked_node(state: ChatState) -> dict[str, Any]:
    """
    Handle cases where the agent cannot proceed.

    Generates a graceful response explaining the limitation.
    """
    reflection = state.get("reflection")
    reason = reflection.reasoning if reflection else "Unable to complete the request"

    prompt = BLOCKED_HANDLER_PROMPT.format(
        query=state.get("user_query", ""),
        reason=reason,
        attempts=format_tool_results(state),
    )

    client = LLMClient()
    response = await client.generate(prompt, model="sonnet")

    return {"final_response": response}


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================


def route_supervisor_decision(state: ChatState) -> Literal["answer", "call_tool", "create_workflow", "clarify"]:
    """Route based on supervisor decision."""
    decision = state.get("current_decision")
    if not decision:
        return "clarify"

    action_map = {
        "ANSWER": "answer",
        "CALL_TOOL": "call_tool",
        "CREATE_WORKFLOW": "create_workflow",
        "CLARIFY": "clarify",
    }
    return action_map.get(decision.action, "clarify")


def route_reflection(state: ChatState) -> Literal["satisfied", "need_more", "blocked"]:
    """Route based on reflection result."""
    reflection = state.get("reflection")
    if not reflection:
        return "satisfied"

    return reflection.assessment
