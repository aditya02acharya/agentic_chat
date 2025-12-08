"""Event type definitions."""

from enum import Enum


class EventType(str, Enum):
    """Types of events emitted during chat processing."""

    THINKING_START = "thinking.start"
    THINKING_UPDATE = "thinking.update"
    THINKING_END = "thinking.end"

    TOOL_START = "tool.start"
    TOOL_PROGRESS = "tool.progress"
    TOOL_RESULT = "tool.result"
    TOOL_ERROR = "tool.error"

    WORKFLOW_START = "workflow.start"
    WORKFLOW_STEP_START = "workflow.step.start"
    WORKFLOW_STEP_COMPLETE = "workflow.step.complete"
    WORKFLOW_COMPLETE = "workflow.complete"
    WORKFLOW_ERROR = "workflow.error"

    RESPONSE_START = "response.start"
    RESPONSE_CHUNK = "response.chunk"
    RESPONSE_DONE = "response.done"

    CLARIFY_REQUEST = "clarify.request"

    ERROR = "error"
