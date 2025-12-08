"""Operator registry with Factory + Registry patterns."""

from typing import Type

from .base import BaseOperator
from ..utils.logging import get_logger

logger = get_logger(__name__)


class OperatorRegistry:
    """
    Registry with factory method for operator creation.

    Design Patterns: Factory + Registry

    Allows dynamic registration and instantiation of operators.
    """

    _operators: dict[str, Type[BaseOperator]] = {}
    _instances: dict[str, BaseOperator] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register operators."""

        def decorator(operator_class: Type[BaseOperator]):
            cls._operators[name] = operator_class
            logger.debug(f"Registered operator: {name}")
            return operator_class

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseOperator:
        """Factory method to create operator instances."""
        if name not in cls._operators:
            raise ValueError(f"Unknown operator: {name}")
        return cls._operators[name](**kwargs)

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseOperator:
        """Get or create a singleton instance of an operator."""
        if name not in cls._instances:
            cls._instances[name] = cls.create(name, **kwargs)
        return cls._instances[name]

    @classmethod
    def list_operators(cls) -> list[dict]:
        """List all registered operators with metadata."""
        result = []
        for name, op_class in cls._operators.items():
            instance = cls.get(name)
            result.append(instance.to_dict())
        return result

    @classmethod
    def get_operator_names(cls) -> list[str]:
        """Get list of registered operator names."""
        return list(cls._operators.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear registry (for testing)."""
        cls._operators.clear()
        cls._instances.clear()
