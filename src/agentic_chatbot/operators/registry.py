"""Operator registry with factory method for dynamic instantiation."""

from typing import Any, Type

from agentic_chatbot.operators.base import BaseOperator
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class OperatorRegistry:
    """
    Registry with factory method for operator creation.

    Design Pattern: Factory + Registry Pattern

    Usage:
        @OperatorRegistry.register("my_operator")
        class MyOperator(BaseOperator):
            ...

        # Later
        operator = OperatorRegistry.create("my_operator")
    """

    _operators: dict[str, Type[BaseOperator]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register operators.

        Args:
            name: Unique operator name

        Returns:
            Decorator function
        """

        def decorator(operator_class: Type[BaseOperator]) -> Type[BaseOperator]:
            if name in cls._operators:
                logger.warning(f"Overwriting operator: {name}")
            cls._operators[name] = operator_class
            logger.debug(f"Registered operator: {name}")
            return operator_class

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseOperator:
        """
        Factory method to create operator instances.

        Args:
            name: Operator name
            **kwargs: Arguments to pass to operator constructor

        Returns:
            Operator instance

        Raises:
            KeyError: If operator not registered
        """
        if name not in cls._operators:
            raise KeyError(f"Operator not found: {name}")
        return cls._operators[name](**kwargs)

    @classmethod
    def get(cls, name: str) -> Type[BaseOperator]:
        """
        Get operator class by name.

        Args:
            name: Operator name

        Returns:
            Operator class

        Raises:
            KeyError: If operator not registered
        """
        if name not in cls._operators:
            raise KeyError(f"Operator not found: {name}")
        return cls._operators[name]

    @classmethod
    def exists(cls, name: str) -> bool:
        """Check if operator is registered."""
        return name in cls._operators

    @classmethod
    def list_operators(cls) -> list[str]:
        """Get list of registered operator names."""
        return list(cls._operators.keys())

    @classmethod
    def get_all_summaries(cls) -> list[dict[str, str]]:
        """Get summaries of all registered operators."""
        summaries = []
        for name, operator_class in cls._operators.items():
            # Create a temporary instance to get summary
            try:
                # Access class attributes directly
                summaries.append(
                    {
                        "name": name,
                        "description": getattr(operator_class, "description", ""),
                        "type": getattr(operator_class, "operator_type", "").value
                        if hasattr(getattr(operator_class, "operator_type", None), "value")
                        else str(getattr(operator_class, "operator_type", "")),
                    }
                )
            except Exception:
                summaries.append({"name": name, "description": "", "type": "unknown"})
        return summaries

    @classmethod
    def get_operators_text(cls) -> str:
        """Get formatted text of all operators for prompts."""
        lines = []
        for summary in cls.get_all_summaries():
            lines.append(f"- {summary['name']}: {summary['description']} (type: {summary['type']})")
        return "\n".join(lines) if lines else "No operators available"

    @classmethod
    def clear(cls) -> None:
        """Clear all registered operators (useful for testing)."""
        cls._operators.clear()
