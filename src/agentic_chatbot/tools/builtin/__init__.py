"""Built-in local tools for self-awareness and introspection."""

from agentic_chatbot.tools.builtin.self_info import SelfInfoTool
from agentic_chatbot.tools.builtin.capabilities import CapabilitiesTool
from agentic_chatbot.tools.builtin.introspection import (
    ListToolsTool,
    ListOperatorsTool,
)

__all__ = [
    "SelfInfoTool",
    "CapabilitiesTool",
    "ListToolsTool",
    "ListOperatorsTool",
]


def register_builtin_tools() -> None:
    """
    Register all built-in local tools.

    Call this during application startup to make built-in tools available.
    """
    # Import triggers registration via decorators
    from agentic_chatbot.tools.builtin import self_info  # noqa: F401
    from agentic_chatbot.tools.builtin import capabilities  # noqa: F401
    from agentic_chatbot.tools.builtin import introspection  # noqa: F401
