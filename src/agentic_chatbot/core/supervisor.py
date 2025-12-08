"""ReACT supervisor logic and decision models."""

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
    """Schema for Supervisor's action decision."""

    action: Literal["ANSWER", "CALL_TOOL", "CREATE_WORKFLOW", "CLARIFY"]
    reasoning: str = Field(description="Explanation of why this action was chosen")

    response: str | None = Field(
        default=None, description="Direct response for ANSWER action"
    )

    operator: str | None = Field(
        default=None, description="Operator name for CALL_TOOL action"
    )
    params: dict[str, Any] | None = Field(
        default=None, description="Parameters for the operator"
    )

    goal: str | None = Field(
        default=None, description="Goal description for CREATE_WORKFLOW"
    )
    steps: list[dict[str, Any]] | None = Field(
        default=None, description="Workflow steps for CREATE_WORKFLOW"
    )

    question: str | None = Field(
        default=None, description="Clarification question for CLARIFY action"
    )


class ReflectionResult(BaseModel):
    """Result of quality evaluation."""

    quality_score: float = Field(ge=0.0, le=1.0, description="Quality score 0.0-1.0")
    is_complete: bool = Field(description="Whether the response is complete")
    issues: list[str] = Field(default_factory=list, description="List of issues found")
    recommendation: Literal["satisfied", "need_more", "blocked"] = Field(
        description="Next action recommendation"
    )
    reasoning: str = Field(description="Explanation of the evaluation")
