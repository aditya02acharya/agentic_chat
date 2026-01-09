"""Core domain modules."""

from agentic_chatbot.core.exceptions import (
    AgenticChatbotError,
    OperatorError,
    WorkflowError,
    MCPError,
    ContextError,
    SupervisorError,
    ValidationError,
)
from agentic_chatbot.core.request_context import RequestContext
from agentic_chatbot.core.supervisor import SupervisorDecision, SupervisorAction

# Query Understanding
from agentic_chatbot.core.query_understanding import (
    QueryUnderstanding,
    QueryIntent,
    QueryComplexity,
    ClarificationRequest,
    ClarificationResponse,
)

# Agent Protocol
from agentic_chatbot.core.agent_protocol import (
    AgentEvent,
    AgentEventType,
    AgentEventBus,
    DelegationContext,
    AgentCapabilities,
    AgentResponse,
    get_event_bus,
)

# Tool Selector
from agentic_chatbot.core.tool_selector import (
    ToolMetadata,
    ToolCandidate,
    ToolSelectionResult,
    ToolSelector,
    get_tool_selector,
)

__all__ = [
    # Exceptions
    "AgenticChatbotError",
    "OperatorError",
    "WorkflowError",
    "MCPError",
    "ContextError",
    "SupervisorError",
    "ValidationError",
    # Context
    "RequestContext",
    # Supervisor
    "SupervisorDecision",
    "SupervisorAction",
    # Query Understanding
    "QueryUnderstanding",
    "QueryIntent",
    "QueryComplexity",
    "ClarificationRequest",
    "ClarificationResponse",
    # Agent Protocol
    "AgentEvent",
    "AgentEventType",
    "AgentEventBus",
    "DelegationContext",
    "AgentCapabilities",
    "AgentResponse",
    "get_event_bus",
    # Tool Selector
    "ToolMetadata",
    "ToolCandidate",
    "ToolSelectionResult",
    "ToolSelector",
    "get_tool_selector",
]
