"""Response subflow for generating and streaming response."""

from pocketflow import AsyncFlow

from agentic_chatbot.nodes.execution.synthesize_node import SynthesizeNode
from agentic_chatbot.nodes.output.write_node import WriteNode
from agentic_chatbot.nodes.output.stream_node import StreamNode


def create_response_subflow() -> AsyncFlow:
    """
    Create response generation subflow.

    Flow:
        [Synthesize?] → Write → Stream

    Returns:
        Configured AsyncFlow for response generation
    """
    # Create nodes
    synthesize = SynthesizeNode()
    write = WriteNode()
    stream = StreamNode()

    # Wire nodes
    synthesize >> write >> stream

    # Create flow
    flow = AsyncFlow(start=synthesize)

    return flow


def create_simple_response_subflow() -> AsyncFlow:
    """
    Create simplified response subflow (write and stream only).

    Flow:
        Write → Stream

    Returns:
        Configured AsyncFlow
    """
    write = WriteNode()
    stream = StreamNode()

    write >> stream

    return AsyncFlow(start=write)
