"""Result storage for the current turn."""

from typing import Any

from pydantic import BaseModel, Field


class ResultStore:
    """
    Stores results from operations during a single turn.

    Features:
    - Key-value storage for named results
    - List storage for sequential results
    - Easy retrieval for synthesis
    """

    def __init__(self):
        self._named: dict[str, Any] = {}
        self._sequential: list[Any] = []

    def store(self, key: str, value: Any) -> None:
        """Store a named result."""
        self._named[key] = value

    def append(self, value: Any) -> None:
        """Append a sequential result."""
        self._sequential.append(value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a named result."""
        return self._named.get(key, default)

    def get_all_named(self) -> dict[str, Any]:
        """Get all named results."""
        return self._named.copy()

    def get_all_sequential(self) -> list[Any]:
        """Get all sequential results."""
        return self._sequential.copy()

    def get_all(self) -> list[Any]:
        """Get all results (named values + sequential)."""
        return list(self._named.values()) + self._sequential

    def clear(self) -> None:
        """Clear all results."""
        self._named.clear()
        self._sequential.clear()

    @property
    def count(self) -> int:
        """Get total result count."""
        return len(self._named) + len(self._sequential)

    @property
    def has_results(self) -> bool:
        """Check if any results exist."""
        return self.count > 0
