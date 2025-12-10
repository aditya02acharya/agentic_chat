"""Tool subflow for executing single tool calls."""

from pocketflow import AsyncFlow

from agentic_chatbot.nodes.context.build_context_node import BuildContextNode
from agentic_chatbot.nodes.execution.tool_node import ExecuteToolNode
from agentic_chatbot.nodes.context.collect_node import CollectResultNode


def create_tool_subflow() -> AsyncFlow:
    """
    Create tool execution subflow.

    Flow:
        BuildContext → ExecuteTool → CollectResult

    Returns:
        Configured AsyncFlow for tool execution
    """
    # Create nodes
    build_context = BuildContextNode()
    execute_tool = ExecuteToolNode(max_retries=2)
    collect_result = CollectResultNode()

    # Wire nodes
    build_context >> execute_tool >> collect_result

    # Create flow
    flow = AsyncFlow(start=build_context)

    return flow
