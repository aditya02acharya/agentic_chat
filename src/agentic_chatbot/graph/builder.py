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
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


def create_chat_graph(
    checkpointer: Any | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
) -> CompiledStateGraph:
    """
    Create the main chat graph.

    This is the top-level graph that orchestrates the entire
    conversation handling process.

    Flow Structure:
        START → initialize → supervisor
            ├── "answer" → write → stream → END
            ├── "call_tool" → execute_tool → reflect
            │                   ├── "satisfied" → synthesize → write → stream → END
            │                   ├── "need_more" → supervisor (loop)
            │                   ├── "blocked" → handle_blocked → write → stream → END
            │                   └── "direct_response" → stream → END (operator bypassed writer)
            ├── "create_workflow" → plan_workflow → execute_workflow → reflect
            │                   ├── "satisfied" → synthesize → write → stream → END
            │                   ├── "need_more" → supervisor (loop)
            │                   ├── "blocked" → handle_blocked → write → stream → END
            │                   └── "direct_response" → stream → END (operator bypassed writer)
            └── "clarify" → clarify → stream → END

    Workflow Execution Features:
        - plan_workflow: LLM creates WorkflowDefinition with steps & dependencies
        - execute_workflow: WorkflowExecutor runs steps with parallel batching
        - Dependency resolution via topological sort
        - Input mapping with {{step_id.output}} templates
        - Per-step event emission for progress tracking

    Direct Response Feature:
        - Operators/tools can send content directly to users (bypass writer)
        - Used for widgets, images, and other rich content
        - When direct response is sent, flow skips synthesize/write nodes
        - Operators set supports_direct_response=True in their capabilities

    Args:
        checkpointer: Optional checkpointer for persistence (MemorySaver, SqliteSaver, etc.)
        interrupt_before: List of node names to interrupt before (for human-in-the-loop)
        interrupt_after: List of node names to interrupt after

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

    # Start -> Initialize -> Supervisor
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "supervisor")

    # Supervisor conditional routing
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor_decision,
        {
            "answer": "write",
            "call_tool": "execute_tool",
            "create_workflow": "plan_workflow",  # Multi-step workflow planning
            "clarify": "clarify",
        },
    )

    # After tool execution -> reflect
    builder.add_edge("execute_tool", "reflect")

    # Workflow: plan -> execute -> reflect
    builder.add_edge("plan_workflow", "execute_workflow")
    builder.add_edge("execute_workflow", "reflect")

    # Reflection conditional routing
    # Note: "direct_response" route is for operators that already sent content
    # directly to the user (bypassing the writer), so we skip to stream
    builder.add_conditional_edges(
        "reflect",
        route_reflection,
        {
            "satisfied": "synthesize",
            "need_more": "supervisor",  # Loop back
            "blocked": "handle_blocked",
            "direct_response": "stream",  # Skip synthesis/write, response already sent
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

    logger.info("Chat graph compiled successfully")

    return graph


def create_chat_graph_with_memory() -> CompiledStateGraph:
    """
    Create chat graph with in-memory checkpointer.

    Useful for development and testing.
    """
    checkpointer = MemorySaver()
    return create_chat_graph(checkpointer=checkpointer)


async def create_chat_graph_with_sqlite(db_path: str = ":memory:") -> CompiledStateGraph:
    """
    Create chat graph with SQLite checkpointer.

    Args:
        db_path: Path to SQLite database file, or ":memory:" for in-memory

    Returns:
        Compiled graph with SQLite persistence
    """
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
            return create_chat_graph(checkpointer=checkpointer)
    except ImportError:
        logger.warning("langgraph-checkpoint-sqlite not installed, using MemorySaver")
        return create_chat_graph_with_memory()


# =============================================================================
# GRAPH EXECUTION HELPERS
# =============================================================================


async def run_chat_graph(
    graph: CompiledGraph,
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
    graph: CompiledGraph,
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
