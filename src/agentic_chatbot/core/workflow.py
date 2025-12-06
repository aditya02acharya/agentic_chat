"""Workflow definition models."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowStepSchema(BaseModel):
    """Schema for workflow step in LLM output."""

    id: str = Field(..., description="Unique step identifier")
    name: str = Field(..., description="Human-readable step name")
    operator: str = Field(..., description="Operator to execute this step")
    input_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Input mappings using {{variable}} syntax",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Step IDs this step depends on",
    )


class WorkflowDefinitionSchema(BaseModel):
    """Schema for workflow definition in LLM output."""

    goal: str = Field(..., description="Overall goal of the workflow")
    steps: list[WorkflowStepSchema] = Field(..., description="Workflow steps")


@dataclass
class WorkflowStep:
    """
    A single step in a workflow.

    Design Pattern: Part of Builder Pattern (created by Supervisor)
    """

    id: str
    name: str
    operator: str
    input_mapping: dict[str, str] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)


@dataclass
class WorkflowDefinition:
    """
    Workflow created by Supervisor for complex tasks.

    Design Pattern: Builder Pattern (created by Supervisor)
    """

    goal: str
    steps: list[WorkflowStep]

    @classmethod
    def from_schema(cls, schema: WorkflowDefinitionSchema) -> "WorkflowDefinition":
        """Create from LLM-generated schema."""
        return cls(
            goal=schema.goal,
            steps=[
                WorkflowStep(
                    id=step.id,
                    name=step.name,
                    operator=step.operator,
                    input_mapping=step.input_mapping,
                    depends_on=step.depends_on,
                )
                for step in schema.steps
            ],
        )


@dataclass
class StepResult:
    """Result from executing a workflow step."""

    step_id: str
    status: WorkflowStatus
    output: Any = None
    error: str | None = None
    duration_ms: float = 0


@dataclass
class WorkflowResult:
    """Result from executing an entire workflow."""

    goal: str
    status: WorkflowStatus
    steps: dict[str, StepResult] = field(default_factory=dict)
    error: str | None = None

    @property
    def is_complete(self) -> bool:
        """Check if workflow completed successfully."""
        return self.status == WorkflowStatus.COMPLETED

    @property
    def failed_steps(self) -> list[str]:
        """Get list of failed step IDs."""
        return [
            step_id
            for step_id, result in self.steps.items()
            if result.status == WorkflowStatus.FAILED
        ]


@dataclass
class WorkflowState:
    """Runtime state of a workflow execution."""

    definition: WorkflowDefinition | None = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    step_results: dict[str, StepResult] = field(default_factory=dict)
    current_step_index: int = 0

    def get_step_output(self, step_id: str) -> Any:
        """Get output from a completed step."""
        if step_id in self.step_results:
            return self.step_results[step_id].output
        return None

    def are_dependencies_met(self, step: WorkflowStep) -> bool:
        """Check if all dependencies for a step are completed."""
        for dep_id in step.depends_on:
            if dep_id not in self.step_results:
                return False
            if self.step_results[dep_id].status != WorkflowStatus.COMPLETED:
                return False
        return True

    def get_ready_steps(self) -> list[WorkflowStep]:
        """Get steps that are ready to execute (dependencies met, not done)."""
        if self.definition is None:
            return []

        ready = []
        for step in self.definition.steps:
            if step.id in self.step_results:
                continue  # Already executed
            if self.are_dependencies_met(step):
                ready.append(step)
        return ready
