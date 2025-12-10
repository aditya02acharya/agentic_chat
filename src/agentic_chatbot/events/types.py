"""Event type enumerations."""

from enum import Enum


class EventType(str, Enum):
    """All event types for SSE streaming."""

    # Supervisor events
    SUPERVISOR_THINKING = "supervisor.thinking"
    SUPERVISOR_DECIDED = "supervisor.decided"

    # Tool/Operator events
    TOOL_START = "tool.start"
    TOOL_PROGRESS = "tool.progress"
    TOOL_CONTENT = "tool.content"  # Multi-modal content (images, widgets)
    TOOL_COMPLETE = "tool.complete"
    TOOL_ERROR = "tool.error"

    # MCP events (internal, mapped to tool events)
    MCP_PROGRESS = "mcp.progress"
    MCP_CONTENT = "mcp.content"
    MCP_ELICITATION = "mcp.elicitation"
    MCP_ERROR = "mcp.error"

    # Workflow events
    WORKFLOW_CREATED = "workflow.created"
    WORKFLOW_STEP_START = "workflow.step.start"
    WORKFLOW_STEP_COMPLETE = "workflow.step.complete"
    WORKFLOW_COMPLETE = "workflow.complete"

    # Response events
    RESPONSE_CHUNK = "response.chunk"
    RESPONSE_DONE = "response.done"

    # User interaction events
    CLARIFY_REQUEST = "clarify.request"
    ERROR = "error"
