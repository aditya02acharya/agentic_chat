"""Domain exceptions for the agentic chatbot."""


class AgenticChatbotError(Exception):
    """Base exception for all agentic chatbot errors."""

    def __init__(self, message: str, recoverable: bool = True):
        super().__init__(message)
        self.recoverable = recoverable


class OperatorError(AgenticChatbotError):
    """Error during operator execution."""

    def __init__(
        self,
        message: str,
        operator_name: str | None = None,
        recoverable: bool = True,
    ):
        super().__init__(message, recoverable)
        self.operator_name = operator_name


class WorkflowError(AgenticChatbotError):
    """Error during workflow execution."""

    def __init__(
        self,
        message: str,
        step_id: str | None = None,
        recoverable: bool = True,
    ):
        super().__init__(message, recoverable)
        self.step_id = step_id


class MCPError(AgenticChatbotError):
    """Error communicating with MCP server."""

    def __init__(
        self,
        message: str,
        server_id: str | None = None,
        tool_name: str | None = None,
        recoverable: bool = True,
    ):
        super().__init__(message, recoverable)
        self.server_id = server_id
        self.tool_name = tool_name


class SupervisorError(AgenticChatbotError):
    """Error in supervisor decision making."""

    def __init__(self, message: str, iteration: int = 0, recoverable: bool = True):
        super().__init__(message, recoverable)
        self.iteration = iteration
