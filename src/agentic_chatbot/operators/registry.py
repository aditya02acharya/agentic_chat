"""Operator registry with factory method for dynamic instantiation."""

from typing import Any, Type

from agentic_chatbot.mcp.models import MessagingCapabilities, OutputDataType
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
    def get_all_summaries(cls) -> list[dict[str, Any]]:
        """Get summaries of all registered operators including messaging capabilities."""
        summaries = []
        for name, operator_class in cls._operators.items():
            try:
                # Access class attributes directly
                output_types = getattr(operator_class, "output_types", [OutputDataType.TEXT])
                summaries.append(
                    {
                        "name": name,
                        "description": getattr(operator_class, "description", ""),
                        "type": getattr(operator_class, "operator_type", "").value
                        if hasattr(getattr(operator_class, "operator_type", None), "value")
                        else str(getattr(operator_class, "operator_type", "")),
                        "messaging": {
                            "output_types": [
                                t.value if hasattr(t, "value") else str(t)
                                for t in output_types
                            ],
                            "supports_progress": getattr(
                                operator_class, "supports_progress", False
                            ),
                            "supports_elicitation": getattr(
                                operator_class, "supports_elicitation", False
                            ),
                            "supports_direct_response": getattr(
                                operator_class, "supports_direct_response", False
                            ),
                            "supports_streaming": getattr(
                                operator_class, "supports_streaming", False
                            ),
                        },
                    }
                )
            except Exception:
                summaries.append({
                    "name": name,
                    "description": "",
                    "type": "unknown",
                    "messaging": {
                        "output_types": ["text"],
                        "supports_progress": False,
                        "supports_elicitation": False,
                        "supports_direct_response": False,
                        "supports_streaming": False,
                    },
                })
        return summaries

    @classmethod
    def get_messaging_capabilities(cls, name: str) -> MessagingCapabilities:
        """
        Get messaging capabilities for a specific operator.

        Args:
            name: Operator name

        Returns:
            MessagingCapabilities instance

        Raises:
            KeyError: If operator not registered
        """
        operator_class = cls.get(name)
        return MessagingCapabilities(
            output_types=getattr(operator_class, "output_types", [OutputDataType.TEXT]),
            supports_progress=getattr(operator_class, "supports_progress", False),
            supports_elicitation=getattr(operator_class, "supports_elicitation", False),
            supports_direct_response=getattr(operator_class, "supports_direct_response", False),
            supports_streaming=getattr(operator_class, "supports_streaming", False),
        )

    @classmethod
    def get_operators_with_capability(
        cls,
        supports_progress: bool | None = None,
        supports_elicitation: bool | None = None,
        supports_direct_response: bool | None = None,
        supports_streaming: bool | None = None,
        output_type: OutputDataType | None = None,
    ) -> list[str]:
        """
        Get operators that match the specified capability requirements.

        Args:
            supports_progress: Filter by progress support
            supports_elicitation: Filter by elicitation support
            supports_direct_response: Filter by direct response support
            supports_streaming: Filter by streaming support
            output_type: Filter by output type support

        Returns:
            List of operator names matching all criteria
        """
        matching = []
        for name, operator_class in cls._operators.items():
            # Check each criterion
            if supports_progress is not None:
                if getattr(operator_class, "supports_progress", False) != supports_progress:
                    continue

            if supports_elicitation is not None:
                if getattr(operator_class, "supports_elicitation", False) != supports_elicitation:
                    continue

            if supports_direct_response is not None:
                if getattr(operator_class, "supports_direct_response", False) != supports_direct_response:
                    continue

            if supports_streaming is not None:
                if getattr(operator_class, "supports_streaming", False) != supports_streaming:
                    continue

            if output_type is not None:
                output_types = getattr(operator_class, "output_types", [OutputDataType.TEXT])
                if output_type not in output_types:
                    continue

            matching.append(name)
        return matching

    @classmethod
    def get_operators_text(cls) -> str:
        """Get formatted text of all operators for prompts including messaging capabilities."""
        lines = []
        for summary in cls.get_all_summaries():
            messaging = summary.get("messaging", {})
            capabilities = []
            if messaging.get("supports_progress"):
                capabilities.append("progress")
            if messaging.get("supports_elicitation"):
                capabilities.append("elicitation")
            if messaging.get("supports_direct_response"):
                capabilities.append("direct_response")
            if messaging.get("supports_streaming"):
                capabilities.append("streaming")

            output_types = messaging.get("output_types", ["text"])
            cap_str = f", capabilities: [{', '.join(capabilities)}]" if capabilities else ""
            out_str = f", outputs: [{', '.join(output_types)}]"

            lines.append(
                f"- {summary['name']}: {summary['description']} "
                f"(type: {summary['type']}{out_str}{cap_str})"
            )
        return "\n".join(lines) if lines else "No operators available"

    @classmethod
    def clear(cls) -> None:
        """Clear all registered operators (useful for testing)."""
        cls._operators.clear()
