"""Context assembler using Builder pattern."""

from typing import Any

from ..operators.context import OperatorContext
from .memory import ConversationMemory
from .results import ResultStore
from .actions import ActionHistory


class ContextAssembler:
    """
    Builds focused context for operators.

    Design Pattern: Builder Pattern

    Assembles only the context elements required by each operator
    based on their context_requirements specification.
    """

    def __init__(
        self,
        query: str,
        memory: ConversationMemory | None = None,
        results: ResultStore | None = None,
        actions: ActionHistory | None = None,
        tool_schemas: dict[str, dict] | None = None,
    ):
        self.query = query
        self.memory = memory or ConversationMemory()
        self.results = results or ResultStore()
        self.actions = actions or ActionHistory()
        self.tool_schemas = tool_schemas or {}
        self._params: dict[str, Any] = {}

    def with_param(self, key: str, value: Any) -> "ContextAssembler":
        """Add a parameter to the context."""
        self._params[key] = value
        return self

    def with_params(self, params: dict[str, Any]) -> "ContextAssembler":
        """Add multiple parameters."""
        self._params.update(params)
        return self

    def build(self, requirements: list[str] | None = None) -> OperatorContext:
        """
        Build operator context based on requirements.

        Args:
            requirements: List of required context elements.
                Supported: "query", "conversation_history", "previous_results",
                          "tools.schema(tool_name)", etc.

        Returns:
            OperatorContext with requested elements
        """
        context = OperatorContext(
            query=self.query,
            params=self._params.copy(),
        )

        if not requirements:
            return context

        for req in requirements:
            if req == "query":
                pass
            elif req == "conversation_history":
                context.conversation_history = self.memory.to_messages_list()
            elif req == "previous_results":
                context.previous_results = self.results.get_all()
            elif req.startswith("tools.schema("):
                tool_name = req[13:-1]
                if tool_name in self.tool_schemas:
                    context.tool_schemas[tool_name] = self.tool_schemas[tool_name]

        return context

    def build_for_operator(self, operator_name: str) -> OperatorContext:
        """Build context for a specific operator using its requirements."""
        from ..operators.registry import OperatorRegistry

        operator = OperatorRegistry.get(operator_name)
        return self.build(operator.context_requirements)
