"""Query Understanding Node.

This node deeply analyzes the user query before any action is taken.
It extracts goals, scope, ecology, and determines if clarification is needed.
"""

from __future__ import annotations

from typing import Any

from agentic_chatbot.core.query_understanding import (
    QueryUnderstanding,
    QueryIntent,
    QueryComplexity,
    QUERY_UNDERSTANDING_PROMPT,
    QUERY_UNDERSTANDING_SYSTEM,
)
from agentic_chatbot.core.tool_selector import get_tool_selector
from agentic_chatbot.graph.state import ChatState
from agentic_chatbot.utils.structured_llm import StructuredLLMCaller
from agentic_chatbot.utils.logging import get_logger
from agentic_chatbot.events.models import SupervisorThinkingEvent


logger = get_logger(__name__)


async def emit_event(state: ChatState, event: Any) -> None:
    """Emit an event to the SSE stream."""
    from agentic_chatbot.graph.state import get_emitter
    emitter = get_emitter(state)
    if emitter:
        await emitter.emit(event)


async def understand_query_node(state: ChatState) -> dict[str, Any]:
    """
    Deeply understand the user's query before taking action.

    This node:
    1. Analyzes the query to extract goals, scope, ecology
    2. Determines if clarification is needed
    3. Suggests tool categories that might be relevant
    4. Assesses query complexity for resource allocation

    Returns:
        State updates including query_understanding and potentially
        needs_clarification flag.
    """
    await emit_event(
        state,
        SupervisorThinkingEvent.create(
            "Understanding your request...",
            request_id=state.get("request_id"),
        ),
    )

    user_query = state.get("user_query", "")
    messages = state.get("messages", [])

    # Build conversation context
    conversation_context = "No previous conversation."
    if len(messages) > 1:
        context_lines = []
        for msg in messages[-6:]:  # Last 6 messages for context
            role = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
            content = msg.content if hasattr(msg, "content") else str(msg)
            context_lines.append(f"{role}: {content[:200]}...")
        conversation_context = "\n".join(context_lines)

    # Get tool categories for context
    tool_selector = get_tool_selector()
    tool_categories = tool_selector.get_category_summary()

    # Build prompt
    prompt = QUERY_UNDERSTANDING_PROMPT.format(
        query=user_query,
        conversation_context=conversation_context,
        tool_categories=tool_categories,
    )

    # Use structured output to extract understanding
    caller = StructuredLLMCaller(max_retries=2)

    try:
        result = await caller.call_with_usage(
            prompt=prompt,
            response_model=QueryUnderstanding,
            system=QUERY_UNDERSTANDING_SYSTEM,
            model="sonnet",  # Fast model for understanding
            enable_thinking=False,  # Don't need extended thinking here
        )

        understanding = result.data
        understanding.raw_query = user_query

        logger.info(
            "Query understood",
            intent=understanding.intent.value,
            complexity=understanding.complexity.value,
            clarity=understanding.clarity_score,
            needs_clarification=understanding.needs_clarification,
            request_id=state.get("request_id"),
        )

        # Determine routing
        result_state: dict[str, Any] = {
            "query_understanding": understanding,
            "token_usage": result.usage,
        }

        # If needs clarification and clarity is low, route to clarification
        if understanding.should_clarify_first():
            result_state["needs_clarification"] = True
            result_state["clarification_questions"] = understanding.clarification_questions

        return result_state

    except Exception as e:
        logger.warning(f"Query understanding failed, proceeding with defaults: {e}")

        # Return default understanding that allows proceeding
        default_understanding = QueryUnderstanding(
            raw_query=user_query,
            reformulated_query=user_query,
            primary_goal="Answer the user's question",
            intent=QueryIntent.INFORMATION_SEEKING,
            complexity=QueryComplexity.MODERATE,
            clarity_score=0.7,
            confidence_score=0.5,
            needs_clarification=False,
        )

        return {"query_understanding": default_understanding}


def route_understanding(state: ChatState) -> str:
    """
    Route based on query understanding.

    Returns:
        "clarify" if clarification needed
        "supervisor" to proceed with action
    """
    needs_clarification = state.get("needs_clarification", False)
    understanding = state.get("query_understanding")

    if needs_clarification:
        # Check if this is truly blocking
        if understanding and understanding.clarity_score < 0.3:
            return "clarify"

    return "supervisor"
