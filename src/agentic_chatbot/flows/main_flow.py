"""Main chat flow orchestrating the complete conversation."""

from pocketflow import AsyncFlow

from agentic_chatbot.nodes.context.init_node import InitializeNode
from agentic_chatbot.nodes.context.fetch_tools_node import FetchToolsNode
from agentic_chatbot.nodes.orchestration.supervisor_node import SupervisorNode
from agentic_chatbot.nodes.orchestration.observe_node import ObserveNode
from agentic_chatbot.nodes.orchestration.reflect_node import ReflectNode
from agentic_chatbot.nodes.orchestration.blocked_node import HandleBlockedNode
from agentic_chatbot.nodes.execution.synthesize_node import SynthesizeNode
from agentic_chatbot.nodes.output.write_node import WriteNode
from agentic_chatbot.nodes.output.stream_node import StreamNode
from agentic_chatbot.nodes.output.clarify_node import ClarifyNode

from agentic_chatbot.flows.tool_subflow import create_tool_subflow
from agentic_chatbot.flows.workflow_subflow import create_simple_workflow_subflow


def create_main_chat_flow() -> AsyncFlow:
    """
    Create the main chat flow.

    This is the top-level graph that orchestrates the entire
    conversation handling process.

    Flow Structure:
        Init → FetchTools → Supervisor
            ├── "answer" → Write → Stream → Done
            ├── "call_tool" → ToolSubFlow → Observe → Reflect
            │                                   ├── "satisfied" → [Synthesize?] → Write → Stream
            │                                   ├── "need_more" → Supervisor (loop)
            │                                   └── "blocked" → HandleBlocked → Write → Stream
            ├── "workflow" → WorkflowSubFlow → Observe → Reflect → ...
            └── "clarify" → Clarify → Stream → Done

    Returns:
        Configured AsyncFlow for chat handling
    """
    # Create main nodes
    init = InitializeNode()
    fetch_tools = FetchToolsNode()
    supervisor = SupervisorNode()
    observe = ObserveNode()
    reflect = ReflectNode()
    handle_blocked = HandleBlockedNode()
    synthesize = SynthesizeNode()
    write = WriteNode()
    stream = StreamNode()
    clarify = ClarifyNode()

    # Create subflows
    tool_subflow = create_tool_subflow()
    workflow_subflow = create_simple_workflow_subflow()

    # Wire the main flow
    # Initial setup
    init >> fetch_tools >> supervisor

    # Supervisor decision routing
    supervisor - "answer" >> write
    supervisor - "call_tool" >> tool_subflow
    supervisor - "create_workflow" >> workflow_subflow
    supervisor - "clarify" >> clarify

    # After tool/workflow execution
    tool_subflow >> observe
    workflow_subflow >> observe

    # Observation leads to reflection
    observe >> reflect

    # Reflection routing
    reflect - "satisfied" >> synthesize
    reflect - "need_more" >> supervisor  # Loop back
    reflect - "blocked" >> handle_blocked

    # Synthesis/blocked handling leads to write
    synthesize >> write
    handle_blocked >> write

    # Write leads to stream (final output)
    write >> stream

    # Clarify also leads to stream
    clarify >> stream

    # Create the flow
    flow = AsyncFlow(start=init)

    return flow


async def run_chat_flow(
    user_query: str,
    conversation_id: str,
    shared_store: dict | None = None,
) -> dict:
    """
    Run the chat flow for a user query.

    Args:
        user_query: User's message
        conversation_id: Conversation identifier
        shared_store: Optional pre-populated shared store

    Returns:
        Final shared store with results
    """
    import asyncio
    from agentic_chatbot.events.models import Event

    # Initialize shared store
    shared = shared_store or {}
    shared["user_query"] = user_query
    shared["conversation_id"] = conversation_id
    shared["request_id"] = f"{conversation_id}_{asyncio.get_event_loop().time()}"
    shared["event_queue"] = asyncio.Queue()

    # Create and run flow
    flow = create_main_chat_flow()
    await flow.run_async(shared)

    return shared
