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


class SupervisorDecision(BaseModel):
    """
    Schema for Supervisor's action decision.

    Used by StructuredLLMCaller to validate LLM output.
    """

    action: Literal["ANSWER", "CALL_TOOL", "CREATE_WORKFLOW", "CLARIFY"]
    reasoning: str = Field(..., description="Explanation of the decision")

    # For ANSWER
    response: str | None = Field(None, description="Direct response for ANSWER action")

    # For CALL_TOOL
    operator: str | None = Field(None, description="Operator name for CALL_TOOL action")
    params: dict[str, Any] | None = Field(None, description="Parameters for the operator")

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
    """State of the supervisor during a turn."""

    iteration: int = 0
    max_iterations: int = 5
    action_history: list[SupervisorActionRecord] = Field(default_factory=list)
    current_decision: SupervisorDecision | None = None
    is_complete: bool = False

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
