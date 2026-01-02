"""Supervisor models and decision structures."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class SupervisorAction(str, Enum):
    """Actions the supervisor can take."""

    ANSWER = "ANSWER"
    CALL_TOOL = "CALL_TOOL"
    CREATE_WORKFLOW = "CREATE_WORKFLOW"
    CLARIFY = "CLARIFY"
    # EXPLORE is handled as CALL_TOOL with discovery tools (browse_tools, search_tools, get_tool_info)


class SupervisorDecision(BaseModel):
    """
    Schema for Supervisor's action decision.

    Used by StructuredLLMCaller to validate LLM output.

    The supervisor operates in a loop:
    1. Analyze query and available information
    2. Decide action (CALL_TOOL for discovery or execution, ANSWER when satisfied, etc.)
    3. Execute and observe results
    4. Loop back to step 1 until satisfied or max iterations

    For tool discovery, use CALL_TOOL with:
    - browse_tools: Explore catalog hierarchically
    - search_tools: Search by keyword
    - get_tool_info: Get full details for a tool
    """

    action: Literal["ANSWER", "CALL_TOOL", "CREATE_WORKFLOW", "CLARIFY"]
    reasoning: str = Field(..., description="Explanation of the decision")

    # For ANSWER
    response: str | None = Field(None, description="Direct response for ANSWER action")

    # For CALL_TOOL (includes discovery tools like browse_tools, search_tools, get_tool_info)
    operator: str | None = Field(None, description="Tool/operator name for CALL_TOOL action")
    params: dict[str, Any] | None = Field(None, description="Parameters for the tool/operator")
    task_description: str | None = Field(
        None,
        description="Clear task description for the operator (operator only sees this, not conversation)"
    )
    task_goal: str | None = Field(None, description="Expected outcome for the task")

    # For CREATE_WORKFLOW
    goal: str | None = Field(None, description="Workflow goal for CREATE_WORKFLOW action")
    steps: list[dict[str, Any]] | None = Field(
        None, description="Workflow steps for CREATE_WORKFLOW action"
    )

    # For CLARIFY
    question: str | None = Field(None, description="Clarification question for CLARIFY action")


class SupervisorActionRecord(BaseModel):
    """Record of an action taken by the supervisor."""

    action: SupervisorAction
    decision: SupervisorDecision
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    iteration: int
    result: dict[str, Any] | None = None
    error: str | None = None


class SupervisorState(BaseModel):
    """State of the supervisor during a turn.

    The supervisor operates in a discovery-execution loop:
    1. Discovery phase: Browse catalog, search tools, get details
    2. Execution phase: Call the actual tools to get data
    3. Reflection phase: Evaluate if we have enough information
    4. Answer phase: Synthesize and respond

    With progressive disclosure, the supervisor may need more iterations
    to first discover the right tools, then execute them.
    """

    iteration: int = 0
    max_iterations: int = 10  # Increased to allow discovery + execution cycles
    action_history: list[SupervisorActionRecord] = Field(default_factory=list)
    current_decision: SupervisorDecision | None = None
    is_complete: bool = False

    # Track discovery vs execution for smarter iteration management
    discovery_actions: int = 0  # browse_tools, search_tools, get_tool_info calls
    execution_actions: int = 0  # actual tool calls for data

    @property
    def actions_this_turn(self) -> list[str]:
        """Get list of action names taken this turn."""
        return [
            f"{record.action.value}: {record.decision.reasoning[:50]}..."
            for record in self.action_history
        ]

    def can_continue(self) -> bool:
        """Check if supervisor can continue iterating."""
        return self.iteration < self.max_iterations and not self.is_complete

    # Discovery tools that don't count against execution limit
    DISCOVERY_TOOLS = {"browse_tools", "search_tools", "get_tool_info", "list_operators", "list_tools"}

    def record_action(
        self,
        decision: SupervisorDecision,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Record an action taken."""
        self.action_history.append(
            SupervisorActionRecord(
                action=SupervisorAction(decision.action),
                decision=decision,
                iteration=self.iteration,
                result=result,
                error=error,
            )
        )
        self.iteration += 1

        # Track discovery vs execution actions
        if decision.action == "CALL_TOOL" and decision.operator:
            tool_name = decision.operator.replace("local:", "")
            if tool_name in self.DISCOVERY_TOOLS:
                self.discovery_actions += 1
            else:
                self.execution_actions += 1

    def can_continue_discovery(self) -> bool:
        """Check if we can continue discovery (more lenient than execution)."""
        # Allow more discovery iterations (exploring the catalog)
        max_discovery = 5
        return self.discovery_actions < max_discovery

    def should_encourage_execution(self) -> bool:
        """Check if we've done enough discovery and should start executing."""
        # After 2+ discovery actions, encourage moving to execution
        return self.discovery_actions >= 2 and self.execution_actions == 0
