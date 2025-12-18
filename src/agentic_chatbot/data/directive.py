"""Directive - what supervisor delegates to operators.

A Directive represents the supervisor's decision and task delegation.
It consolidates SupervisorDecision and related types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from agentic_chatbot.data.execution import TaskInfo


class DirectiveType(str, Enum):
    """Types of directives the supervisor can issue."""

    ANSWER = "ANSWER"  # Respond directly to user
    CALL_OPERATOR = "CALL_OPERATOR"  # Execute single operator
    CREATE_WORKFLOW = "CREATE_WORKFLOW"  # Create multi-step plan
    CLARIFY = "CLARIFY"  # Ask user for clarification


@dataclass
class Directive:
    """
    Supervisor's decision and task delegation.

    Replaces SupervisorDecision with a cleaner structure.

    Usage:
        # Direct answer
        directive = Directive.answer("Python is a programming language.")

        # Call operator
        directive = Directive.call_operator(
            operator="web_search",
            params={"query": "python async tutorial"},
            task=TaskInfo(
                description="Search for Python async tutorials",
                goal="Find comprehensive async/await guides",
            ),
        )

        # Create workflow
        directive = Directive.workflow(
            goal="Research and summarize topic",
            steps=[
                {"operator": "web_search", "params": {...}},
                {"operator": "summarizer", "params": {...}},
            ],
        )

        # Clarify
        directive = Directive.clarify("What programming language are you asking about?")
    """

    directive_type: DirectiveType
    reasoning: str

    # For ANSWER
    response: str | None = None

    # For CALL_OPERATOR
    operator: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    task: TaskInfo | None = None

    # For CREATE_WORKFLOW
    goal: str | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)

    # For CLARIFY
    question: str | None = None

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    iteration: int = 0

    # Properties
    @property
    def action(self) -> str:
        """Get action string (for compatibility)."""
        return self.directive_type.value

    @property
    def is_terminal(self) -> bool:
        """Check if this directive ends the turn."""
        return self.directive_type in (DirectiveType.ANSWER, DirectiveType.CLARIFY)

    # Factory methods
    @classmethod
    def answer(cls, response: str, reasoning: str = "") -> "Directive":
        """Create direct answer directive."""
        return cls(
            directive_type=DirectiveType.ANSWER,
            reasoning=reasoning or "Answering directly",
            response=response,
        )

    @classmethod
    def call_operator(
        cls,
        operator: str,
        params: dict[str, Any] | None = None,
        task: TaskInfo | None = None,
        reasoning: str = "",
    ) -> "Directive":
        """Create call operator directive."""
        return cls(
            directive_type=DirectiveType.CALL_OPERATOR,
            reasoning=reasoning or f"Calling {operator}",
            operator=operator,
            params=params or {},
            task=task,
        )

    @classmethod
    def workflow(
        cls,
        goal: str,
        steps: list[dict[str, Any]] | None = None,
        reasoning: str = "",
    ) -> "Directive":
        """Create workflow directive."""
        return cls(
            directive_type=DirectiveType.CREATE_WORKFLOW,
            reasoning=reasoning or "Creating multi-step plan",
            goal=goal,
            steps=steps or [],
        )

    @classmethod
    def clarify(cls, question: str, reasoning: str = "") -> "Directive":
        """Create clarify directive."""
        return cls(
            directive_type=DirectiveType.CLARIFY,
            reasoning=reasoning or "Need clarification",
            question=question,
        )

    # Conversion methods
    def to_execution_input(self, query: str) -> "ExecutionInput":
        """Convert to ExecutionInput for operator execution."""
        from agentic_chatbot.data.execution import ExecutionInput

        return ExecutionInput(
            query=query,
            task=self.task,
            extra=self.params,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "directive_type": self.directive_type.value,
            "reasoning": self.reasoning,
            "response": self.response,
            "operator": self.operator,
            "params": self.params,
            "task": {
                "description": self.task.description,
                "goal": self.task.goal,
                "scope": self.task.scope,
                "constraints": self.task.constraints,
                "original_query": self.task.original_query,
            }
            if self.task
            else None,
            "goal": self.goal,
            "steps": self.steps,
            "question": self.question,
            "timestamp": self.timestamp.isoformat(),
            "iteration": self.iteration,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Directive":
        """Deserialize from dictionary."""
        task_data = d.get("task")
        task = None
        if task_data:
            task = TaskInfo(
                description=task_data["description"],
                goal=task_data["goal"],
                scope=task_data.get("scope", ""),
                constraints=task_data.get("constraints", []),
                original_query=task_data.get("original_query", ""),
            )

        return cls(
            directive_type=DirectiveType(d["directive_type"]),
            reasoning=d["reasoning"],
            response=d.get("response"),
            operator=d.get("operator"),
            params=d.get("params", {}),
            task=task,
            goal=d.get("goal"),
            steps=d.get("steps", []),
            question=d.get("question"),
            iteration=d.get("iteration", 0),
        )


@dataclass
class DirectiveRecord:
    """
    Record of a directive and its outcome.

    Used for tracking action history during a turn.
    """

    directive: Directive
    outcome: "DirectiveOutcome | None" = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Check if directive execution succeeded."""
        return self.outcome is not None and self.error is None


@dataclass
class DirectiveOutcome:
    """
    Outcome of executing a directive.

    Contains the result summary (not raw data).
    """

    summary: str  # Brief description of what happened
    has_data: bool = True  # Did we get useful data?
    next_action_hint: str = ""  # Suggestion for next action

    # Metrics
    duration_ms: float = 0.0
    tokens_used: int = 0
