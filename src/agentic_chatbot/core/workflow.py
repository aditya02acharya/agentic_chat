"""Workflow definition models."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    """Status of a workflow step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStep(BaseModel):
    """Definition of a single workflow step."""

    id: str = Field(description="Unique step identifier")
    name: str = Field(description="Human-readable step name")
    operator: str = Field(description="Operator to execute")
    params: dict[str, Any] = Field(default_factory=dict, description="Operator params")
    dependencies: list[str] = Field(
        default_factory=list, description="IDs of steps this depends on"
    )
    description: str | None = Field(default=None, description="Step description")


class WorkflowDefinition(BaseModel):
    """Complete workflow definition."""

    goal: str = Field(description="Overall goal of the workflow")
    steps: list[WorkflowStep] = Field(description="Steps to execute")
    parallel_groups: list[list[str]] | None = Field(
        default=None, description="Groups of step IDs that can run in parallel"
    )


class StepResult(BaseModel):
    """Result from executing a workflow step."""

    step_id: str
    status: StepStatus
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0


class WorkflowResult(BaseModel):
    """Complete workflow execution result."""

    definition: WorkflowDefinition
    step_results: dict[str, StepResult] = Field(default_factory=dict)
    status: str = "pending"
    total_duration_ms: float = 0.0
