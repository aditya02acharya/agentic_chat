"""Main chat flow orchestrating the complete conversation.

DEPRECATED: This module is kept for backwards compatibility.
The new implementation uses LangGraph and is in the graph/ module.

Use:
    from agentic_chatbot.graph import create_chat_graph

instead of:
    from agentic_chatbot.flows.main_flow import create_main_chat_flow
"""

import asyncio
from typing import Any

from langgraph.graph.graph import CompiledGraph

from agentic_chatbot.graph import create_chat_graph
from agentic_chatbot.graph.state import create_initial_state, ChatState


def create_main_chat_flow() -> CompiledGraph:
    """
    Create the main chat flow using LangGraph.

    DEPRECATED: Use create_chat_graph() from agentic_chatbot.graph instead.

    This function is kept for backwards compatibility.

    Returns:
        Compiled LangGraph StateGraph
    """
    return create_chat_graph()


async def run_chat_flow(
    user_query: str,
    conversation_id: str,
    shared_store: dict[str, Any] | None = None,
) -> ChatState:
    """
    Run the chat flow for a user query.

    DEPRECATED: Use the graph module directly:
        graph = create_chat_graph()
        state = create_initial_state(...)
        result = await graph.ainvoke(state, config)

    Args:
        user_query: User's message
        conversation_id: Conversation identifier
        shared_store: Optional pre-populated shared store (ignored in new implementation)

    Returns:
        Final state with results
    """
    request_id = f"{conversation_id}_{asyncio.get_event_loop().time()}"

    # Create initial state
    initial_state = create_initial_state(
        user_query=user_query,
        conversation_id=conversation_id,
        request_id=request_id,
    )

    # Create and run graph
    graph = create_chat_graph()
    config = {
        "configurable": {
            "thread_id": conversation_id,
        }
    }

    final_state = await graph.ainvoke(initial_state, config)
    return final_state
