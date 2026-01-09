"""LangGraph builder for the agentic chatbot.

Creates the StateGraph with all nodes and conditional edges.
Supports both in-memory and persistent checkpointers.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from agentic_chatbot.graph.state import ChatState
from agentic_chatbot.graph.nodes import (
    # Node functions
    initialize_node,
    supervisor_node,
    execute_tool_node,
    plan_workflow_node,
    execute_workflow_node,
    reflect_node,
    synthesize_node,
    write_node,
    stream_node,
    clarify_node,
    handle_blocked_node,
    # Routing functions
    route_supervisor_decision,
    route_reflection,
)
from agentic_chatbot.nodes.orchestration.understand_node import (
    understand_query_node,
    route_understanding,
)
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


def create_chat_graph(
    checkpointer: Any | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    enable_understanding: bool = True,
) -> CompiledStateGraph:
    """
    Create the main chat graph.

    This is the top-level graph that orchestrates the entire
    conversation handling process.

    Flow Structure (with understanding enabled):
        START → initialize → understand → [route]
            ├── "clarify" → clarify → stream → END (if vague query)
            └── "supervisor" → supervisor
                ├── "answer" → write → stream → END
                ├── "call_tool" → execute_tool → reflect
                │                   ├── "satisfied" → synthesize → write → stream → END
                │                   ├── "need_more" → supervisor (loop)
                │                   ├── "blocked" → handle_blocked → write → stream → END
                │                   └── "direct_response" → stream → END
                ├── "create_workflow" → plan_workflow → execute_workflow → reflect
                │                   ├── "satisfied" → synthesize → write → stream → END
                │                   ├── "need_more" → supervisor (loop)
                │                   ├── "blocked" → handle_blocked → write → stream → END
                │                   └── "direct_response" → stream → END
                └── "clarify" → clarify → stream → END

    Key Features:
        - Query Understanding: Deep analysis before action (goals, scope, ecology)
        - Tool Selection: Efficient filtering to top N candidate tools
        - Event-Driven Communication: Agents communicate via events
        - Delegation Control: Supervisor controls thinking budgets for delegates

    Args:
        checkpointer: Optional checkpointer for persistence
        interrupt_before: List of node names to interrupt before
        interrupt_after: List of node names to interrupt after
        enable_understanding: Whether to include query understanding stage

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create the StateGraph with our state schema
    builder = StateGraph(ChatState)

    # -------------------------------------------------------------------------
    # ADD NODES
    # -------------------------------------------------------------------------

    # Initialization
    builder.add_node("initialize", initialize_node)

    # Query Understanding (optional but recommended)
    if enable_understanding:
        builder.add_node("understand", understand_query_node)

    # Orchestration
    builder.add_node("supervisor", supervisor_node)

    # Execution - Single tool
    builder.add_node("execute_tool", execute_tool_node)

    # Execution - Multi-step workflow
    builder.add_node("plan_workflow", plan_workflow_node)
    builder.add_node("execute_workflow", execute_workflow_node)

    # Reflection & Synthesis
    builder.add_node("reflect", reflect_node)
    builder.add_node("synthesize", synthesize_node)

    # Output
    builder.add_node("write", write_node)
    builder.add_node("stream", stream_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("handle_blocked", handle_blocked_node)

    # -------------------------------------------------------------------------
    # ADD EDGES
    # -------------------------------------------------------------------------

    # Start -> Initialize
    builder.add_edge(START, "initialize")

    if enable_understanding:
        # Initialize -> Understand -> Route
        builder.add_edge("initialize", "understand")
        builder.add_conditional_edges(
            "understand",
            route_understanding,
            {
                "clarify": "clarify",  # Query too vague, ask for clarification
                "supervisor": "supervisor",  # Proceed with action
            },
        )
    else:
        # Initialize -> Supervisor (skip understanding)
        builder.add_edge("initialize", "supervisor")

    # Supervisor conditional routing
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor_decision,
        {
            "answer": "write",
            "call_tool": "execute_tool",
            "create_workflow": "plan_workflow",
            "clarify": "clarify",
        },
    )

    # After tool execution -> reflect
    builder.add_edge("execute_tool", "reflect")

    # Workflow: plan -> execute -> reflect
    builder.add_edge("plan_workflow", "execute_workflow")
    builder.add_edge("execute_workflow", "reflect")

    # Reflection conditional routing
    builder.add_conditional_edges(
        "reflect",
        route_reflection,
        {
            "satisfied": "synthesize",
            "need_more": "supervisor",
            "blocked": "handle_blocked",
            "direct_response": "stream",
        },
    )

    # Synthesis -> Write
    builder.add_edge("synthesize", "write")

    # Handle blocked -> Write
    builder.add_edge("handle_blocked", "write")

    # Write -> Stream -> END
    builder.add_edge("write", "stream")
    builder.add_edge("stream", END)

    # Clarify -> Stream -> END
    builder.add_edge("clarify", "stream")

    # -------------------------------------------------------------------------
    # COMPILE GRAPH
    # -------------------------------------------------------------------------

    compile_kwargs: dict[str, Any] = {}

    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before

    if interrupt_after:
        compile_kwargs["interrupt_after"] = interrupt_after

    graph = builder.compile(**compile_kwargs)

    logger.info(
        "Chat graph compiled successfully",
        understanding_enabled=enable_understanding,
    )

    return graph


def create_chat_graph_with_memory(enable_understanding: bool = True) -> CompiledStateGraph:
    """
    Create chat graph with in-memory checkpointer.

    Useful for development and testing.
    """
    checkpointer = MemorySaver()
    return create_chat_graph(
        checkpointer=checkpointer,
        enable_understanding=enable_understanding,
    )


async def create_chat_graph_with_sqlite(
    db_path: str = ":memory:",
    enable_understanding: bool = True,
) -> CompiledStateGraph:
    """
    Create chat graph with SQLite checkpointer.

    Args:
        db_path: Path to SQLite database file, or ":memory:" for in-memory
        enable_understanding: Whether to include query understanding stage

    Returns:
        Compiled graph with SQLite persistence
    """
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        checkpointer = AsyncSqliteSaver.from_conn_string(db_path)
        await checkpointer.setup()
        return create_chat_graph(
            checkpointer=checkpointer,
            enable_understanding=enable_understanding,
        )
    except ImportError:
        logger.warning("langgraph-checkpoint-sqlite not installed, using MemorySaver")
        return create_chat_graph_with_memory(enable_understanding=enable_understanding)


async def create_chat_graph_with_sqlite_managed(
    db_path: str = ":memory:",
    enable_understanding: bool = True,
) -> tuple[CompiledStateGraph, Any]:
    """
    Create chat graph with SQLite checkpointer, returning both for lifecycle management.

    Args:
        db_path: Path to SQLite database file, or ":memory:" for in-memory
        enable_understanding: Whether to include query understanding stage

    Returns:
        Tuple of (compiled graph, checkpointer) - caller must close checkpointer
    """
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        checkpointer = AsyncSqliteSaver.from_conn_string(db_path)
        await checkpointer.setup()
        return create_chat_graph(
            checkpointer=checkpointer,
            enable_understanding=enable_understanding,
        ), checkpointer
    except ImportError:
        logger.warning("langgraph-checkpoint-sqlite not installed, using MemorySaver")
        checkpointer = MemorySaver()
        return create_chat_graph(
            checkpointer=checkpointer,
            enable_understanding=enable_understanding,
        ), checkpointer


# =============================================================================
# GRAPH EXECUTION HELPERS
# =============================================================================


async def run_chat_graph(
    graph: CompiledStateGraph,
    state: ChatState,
    config: dict[str, Any] | None = None,
) -> ChatState:
    """
    Run the chat graph and return final state.

    Args:
        graph: Compiled chat graph
        state: Initial state
        config: Optional config with thread_id for persistence

    Returns:
        Final state after graph execution
    """
    config = config or {}

    # Ensure thread_id for checkpointing
    if "configurable" not in config:
        config["configurable"] = {}
    if "thread_id" not in config["configurable"]:
        config["configurable"]["thread_id"] = state.get("conversation_id", "default")

    result = await graph.ainvoke(state, config)
    return result


async def stream_chat_graph(
    graph: CompiledStateGraph,
    state: ChatState,
    config: dict[str, Any] | None = None,
):
    """
    Stream graph execution, yielding state updates.

    Args:
        graph: Compiled chat graph
        state: Initial state
        config: Optional config with thread_id

    Yields:
        State updates from each node
    """
    config = config or {}

    if "configurable" not in config:
        config["configurable"] = {}
    if "thread_id" not in config["configurable"]:
        config["configurable"]["thread_id"] = state.get("conversation_id", "default")

    async for event in graph.astream(state, config, stream_mode="updates"):
        yield event
