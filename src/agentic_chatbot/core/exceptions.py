"""Core domain exceptions."""


class AgenticChatbotError(Exception):
    """Base exception for all agentic chatbot errors."""

    pass


class OperatorError(AgenticChatbotError):
    """Error in operator execution."""

    def __init__(self, message: str, operator_name: str | None = None):
        super().__init__(message)
        self.operator_name = operator_name


class WorkflowError(AgenticChatbotError):
    """Error in workflow execution."""

    def __init__(self, message: str, step_id: str | None = None):
        super().__init__(message)
        self.step_id = step_id


class MCPError(AgenticChatbotError):
    """Error in MCP communication."""

    def __init__(
        self,
        message: str,
        server_id: str | None = None,
        tool_name: str | None = None,
        error_type: str = "general",
    ):
        super().__init__(message)
        self.server_id = server_id
        self.tool_name = tool_name
        self.error_type = error_type


class ContextError(AgenticChatbotError):
    """Error in context assembly or management."""

    def __init__(self, message: str, requirement: str | None = None):
        super().__init__(message)
        self.requirement = requirement


class SupervisorError(AgenticChatbotError):
    """Error in supervisor decision-making."""

    def __init__(self, message: str, iteration: int | None = None):
        super().__init__(message)
        self.iteration = iteration


class ValidationError(AgenticChatbotError):
    """Validation error for inputs or outputs."""

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field


class ToolNotFoundError(MCPError):
    """Requested tool not found in registry."""

    def __init__(self, tool_name: str):
        super().__init__(f"Tool '{tool_name}' not found", tool_name=tool_name)


class ServerNotFoundError(MCPError):
    """Requested MCP server not found."""

    def __init__(self, server_id: str):
        super().__init__(f"Server '{server_id}' not found", server_id=server_id)


class TransientError(MCPError):
    """Transient error that may be retried."""

    pass


class ElicitationTimeoutError(MCPError):
    """Timeout waiting for user response to elicitation."""

    def __init__(self, server_id: str, tool_name: str, timeout: float):
        super().__init__(
            f"Timeout ({timeout}s) waiting for elicitation response",
            server_id=server_id,
            tool_name=tool_name,
            error_type="timeout",
        )
