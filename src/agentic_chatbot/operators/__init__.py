"""Operators module - Strategy pattern for interchangeable algorithms."""

from agentic_chatbot.operators.base import BaseOperator, OperatorType
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.operators.context import (
    OperatorContext,
    OperatorResult,
    MessagingContext,
)

__all__ = [
    "BaseOperator",
    "OperatorType",
    "OperatorRegistry",
    "OperatorContext",
    "OperatorResult",
    "MessagingContext",
]
