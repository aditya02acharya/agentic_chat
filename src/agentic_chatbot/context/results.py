"""Result store for managing operator and workflow outputs."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agentic_chatbot.operators.context import OperatorResult


class StoredResult(BaseModel):
    """A stored result with metadata."""

    key: str = Field(..., description="Result identifier")
    result: OperatorResult = Field(..., description="The operator result")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = Field("", description="Source (tool name, step id, etc.)")


class ResultStore:
    """
    Store for operator results during a turn.

    Manages results from tool calls and workflow steps,
    making them available for context assembly and synthesis.
    """

    def __init__(self):
        """Initialize result store."""
        self._results: dict[str, StoredResult] = {}
        self._tool_outputs: list[OperatorResult] = []

    async def get(self, method: str) -> Any:
        """
        Get results based on method string.

        Supports:
        - "step(N)" or "step(id)": Output from specific workflow step
        - "all": All results this turn
        - "tool(name)": Results from specific tool
        - "latest": Most recent result

        Args:
            method: Method string specifying what to retrieve

        Returns:
            Requested result data
        """
        if method.startswith("step("):
            # Parse step ID from "step(2)" or "step(my_step)"
            step_id = method[5:-1]
            return self.get_step_result(step_id)
        elif method == "all":
            return self.get_all()
        elif method.startswith("tool("):
            tool_name = method[5:-1]
            return self.get_by_source(tool_name)
        elif method == "latest":
            return self.get_latest()
        else:
            return self._results.get(method)

    def store(
        self,
        key: str,
        result: OperatorResult,
        source: str = "",
    ) -> None:
        """
        Store a result.

        Args:
            key: Unique identifier for this result
            result: The operator result
            source: Source identifier (tool name, step id)
        """
        self._results[key] = StoredResult(
            key=key,
            result=result,
            source=source or key,
        )
        self._tool_outputs.append(result)

    def store_tool_output(self, result: OperatorResult) -> None:
        """Store a tool output (convenience method)."""
        key = f"tool_{len(self._tool_outputs)}"
        self.store(key, result, source=result.metadata.get("tool", key))

    def store_step_result(self, step_id: str, result: OperatorResult) -> None:
        """Store a workflow step result."""
        self.store(f"step_{step_id}", result, source=step_id)

    def get_step_result(self, step_id: str) -> OperatorResult | None:
        """Get result for a specific workflow step."""
        key = f"step_{step_id}" if not step_id.startswith("step_") else step_id
        stored = self._results.get(key)
        return stored.result if stored else None

    def get_by_source(self, source: str) -> list[OperatorResult]:
        """Get all results from a specific source."""
        return [
            stored.result
            for stored in self._results.values()
            if stored.source == source
        ]

    def get_all(self) -> dict[str, Any]:
        """Get all results as a dictionary."""
        return {
            key: stored.result.output
            for key, stored in self._results.items()
        }

    def get_all_results(self) -> list[OperatorResult]:
        """Get all operator results."""
        return [stored.result for stored in self._results.values()]

    def get_latest(self) -> OperatorResult | None:
        """Get the most recent result."""
        if not self._results:
            return None
        # Get by timestamp
        latest = max(self._results.values(), key=lambda x: x.timestamp)
        return latest.result

    def get_tool_outputs(self) -> list[OperatorResult]:
        """Get all tool outputs in order."""
        return self._tool_outputs.copy()

    def clear(self) -> None:
        """Clear all stored results."""
        self._results.clear()
        self._tool_outputs.clear()

    def __len__(self) -> int:
        """Return number of stored results."""
        return len(self._results)

    @property
    def has_results(self) -> bool:
        """Check if any results are stored."""
        return len(self._results) > 0

    def format_for_prompt(self) -> str:
        """Format all results for inclusion in prompts."""
        if not self._results:
            return "No results collected."

        parts = []
        for key, stored in self._results.items():
            output = stored.result.text_output
            parts.append(f"### {key} (from {stored.source})\n{output[:1000]}")

        return "\n\n".join(parts)
