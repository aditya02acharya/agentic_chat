"""Action history for tracking supervisor actions."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agentic_chatbot.core.supervisor import SupervisorAction, SupervisorDecision


class ActionRecord(BaseModel):
    """Record of a single action."""

    action_type: str = Field(..., description="Type of action taken")
    description: str = Field("", description="Brief description")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
    success: bool = Field(True, description="Whether action succeeded")
    error: str | None = Field(None, description="Error message if failed")


class ActionHistory:
    """
    Tracks actions taken during a supervisor turn.

    Used for:
    - Context assembly (showing what's been tried)
    - Reflection (evaluating progress)
    - Loop detection (avoiding repeated failures)
    """

    def __init__(self):
        """Initialize action history."""
        self._actions: list[ActionRecord] = []
        self._turn_start: datetime = datetime.utcnow()

    async def get(self, method: str) -> Any:
        """
        Get action history based on method string.

        Supports:
        - "this_turn": Actions taken this turn
        - "all": All actions
        - "failed": Only failed actions
        - "count": Number of actions

        Args:
            method: Method string specifying what to retrieve

        Returns:
            Requested action data
        """
        if method == "this_turn":
            return self.get_this_turn()
        elif method == "all":
            return self.get_all()
        elif method == "failed":
            return self.get_failed()
        elif method == "count":
            return len(self._actions)
        else:
            return None

    def record(
        self,
        action_type: str,
        description: str = "",
        success: bool = True,
        error: str | None = None,
        **metadata: Any,
    ) -> None:
        """
        Record an action.

        Args:
            action_type: Type of action (e.g., "CALL_TOOL", "CREATE_WORKFLOW")
            description: Brief description of what was done
            success: Whether the action succeeded
            error: Error message if failed
            **metadata: Additional metadata
        """
        self._actions.append(
            ActionRecord(
                action_type=action_type,
                description=description,
                success=success,
                error=error,
                metadata=metadata,
            )
        )

    def record_decision(
        self,
        decision: SupervisorDecision,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """
        Record a supervisor decision.

        Args:
            decision: The supervisor's decision
            success: Whether execution succeeded
            error: Error message if failed
        """
        description = decision.reasoning[:100] if decision.reasoning else ""

        metadata = {}
        if decision.operator:
            metadata["operator"] = decision.operator
        if decision.goal:
            metadata["goal"] = decision.goal

        self.record(
            action_type=decision.action,
            description=description,
            success=success,
            error=error,
            **metadata,
        )

    def get_this_turn(self) -> list[str]:
        """Get formatted list of actions this turn."""
        return [
            f"{a.action_type}: {a.description}" + (f" (FAILED: {a.error})" if not a.success else "")
            for a in self._actions
        ]

    def get_all(self) -> list[ActionRecord]:
        """Get all action records."""
        return self._actions.copy()

    def get_failed(self) -> list[ActionRecord]:
        """Get only failed actions."""
        return [a for a in self._actions if not a.success]

    def get_successful(self) -> list[ActionRecord]:
        """Get only successful actions."""
        return [a for a in self._actions if a.success]

    def has_action_type(self, action_type: str) -> bool:
        """Check if an action type was attempted this turn."""
        return any(a.action_type == action_type for a in self._actions)

    def count_action_type(self, action_type: str) -> int:
        """Count how many times an action type was attempted."""
        return sum(1 for a in self._actions if a.action_type == action_type)

    def start_new_turn(self) -> None:
        """Start a new turn (clears history)."""
        self._actions.clear()
        self._turn_start = datetime.utcnow()

    def clear(self) -> None:
        """Clear all action history."""
        self._actions.clear()

    def __len__(self) -> int:
        """Return number of actions."""
        return len(self._actions)

    def format_for_prompt(self) -> str:
        """Format action history for inclusion in prompts."""
        if not self._actions:
            return "No actions taken yet."

        lines = []
        for i, action in enumerate(self._actions, 1):
            status = "✓" if action.success else "✗"
            line = f"{i}. [{status}] {action.action_type}: {action.description}"
            if action.error:
                line += f"\n   Error: {action.error}"
            lines.append(line)

        return "\n".join(lines)
