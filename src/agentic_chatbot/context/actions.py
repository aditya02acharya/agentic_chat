"""Action history tracking."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..core.supervisor import SupervisorAction


class ActionRecord(BaseModel):
    """Record of a supervisor action."""

    action: SupervisorAction
    operator: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    result_summary: str | None = None
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0


class ActionHistory:
    """
    Tracks actions taken during a supervisor session.

    Features:
    - Ordered action log
    - Summary generation
    - Action filtering by type
    """

    def __init__(self):
        self._actions: list[ActionRecord] = []

    def record(
        self,
        action: SupervisorAction,
        operator: str | None = None,
        params: dict[str, Any] | None = None,
        result_summary: str | None = None,
        success: bool = True,
        duration_ms: float = 0.0,
    ) -> None:
        """Record an action."""
        self._actions.append(
            ActionRecord(
                action=action,
                operator=operator,
                params=params or {},
                result_summary=result_summary,
                success=success,
                duration_ms=duration_ms,
            )
        )

    def get_all(self) -> list[ActionRecord]:
        """Get all recorded actions."""
        return self._actions.copy()

    def get_by_type(self, action_type: SupervisorAction) -> list[ActionRecord]:
        """Get actions of a specific type."""
        return [a for a in self._actions if a.action == action_type]

    def get_summary(self) -> str:
        """Get a text summary of actions taken."""
        if not self._actions:
            return "No actions taken yet."

        lines = []
        for i, action in enumerate(self._actions, 1):
            line = f"{i}. {action.action.value}"
            if action.operator:
                line += f" ({action.operator})"
            if action.result_summary:
                line += f": {action.result_summary[:100]}"
            lines.append(line)
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear action history."""
        self._actions.clear()

    @property
    def count(self) -> int:
        """Get number of recorded actions."""
        return len(self._actions)

    @property
    def last_action(self) -> ActionRecord | None:
        """Get the most recent action."""
        return self._actions[-1] if self._actions else None
