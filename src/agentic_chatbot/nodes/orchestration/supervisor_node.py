"""Supervisor node implementing ReACT pattern."""

from typing import Any

from ..base import AsyncBaseNode
from ...core.request_context import RequestContext
from ...core.supervisor import SupervisorDecision, SupervisorAction
from ...config.prompts import SUPERVISOR_SYSTEM_PROMPT
from ...operators.registry import OperatorRegistry
from ...events.types import EventType
from ...utils.llm import LLMClient
from ...utils.structured_llm import StructuredLLMCaller, StructuredOutputError
from ...utils.logging import get_logger

logger = get_logger(__name__)


class SupervisorNode(AsyncBaseNode):
    """
    ReACT supervisor node that decides on actions.

    Implements the ReACT loop:
    - THINK: Analyze the query
    - REASON: Determine what's needed
    - PLAN: Choose action type
    - ACT: Return action decision
    """

    name = "supervisor"

    def __init__(self, ctx: RequestContext):
        super().__init__(ctx)
        self._llm = LLMClient()
        self._structured_llm = StructuredLLMCaller(self._llm)

    async def execute(self, shared: dict[str, Any]) -> str:
        await self.emit_event(EventType.THINKING_START, {"phase": "supervisor"})

        query = shared.get("query", self.ctx.user_query)
        iteration = shared.get("iteration", 0)
        max_iterations = shared.get("max_iterations", 5)

        if iteration >= max_iterations:
            logger.warning(f"Max iterations ({max_iterations}) reached")
            shared["decision"] = SupervisorDecision(
                action="ANSWER",
                reasoning="Maximum iterations reached, providing best available response",
                response="I apologize, but I've reached my limit for this request. Here's what I found so far.",
            )
            return "answer"

        shared["iteration"] = iteration + 1

        operators = OperatorRegistry.list_operators()
        operator_names = ", ".join(op["name"] for op in operators)

        action_history = shared.get("action_history", "")
        previous_results = shared.get("previous_results", [])

        prompt = self._build_prompt(
            query=query,
            operators=operator_names,
            action_history=action_history,
            previous_results=previous_results,
        )

        system = SUPERVISOR_SYSTEM_PROMPT.format(
            available_operators=operator_names
        )

        try:
            decision = await self._structured_llm.call(
                prompt=prompt,
                response_model=SupervisorDecision,
                system=system,
                model="sonnet",
            )
            shared["decision"] = decision

            await self.emit_event(
                EventType.THINKING_END,
                {"action": decision.action, "reasoning": decision.reasoning},
            )

            return decision.action.lower()

        except StructuredOutputError as e:
            logger.error(f"Supervisor decision failed: {e.attempts}")
            shared["decision"] = SupervisorDecision(
                action="CLARIFY",
                reasoning="Unable to determine appropriate action",
                question="Could you please rephrase your request?",
            )
            return "clarify"

    def _build_prompt(
        self,
        query: str,
        operators: str,
        action_history: str,
        previous_results: list,
    ) -> str:
        prompt = f"User Query: {query}\n\n"

        if action_history:
            prompt += f"Actions Taken So Far:\n{action_history}\n\n"

        if previous_results:
            results_text = "\n".join(str(r)[:500] for r in previous_results[-3:])
            prompt += f"Previous Results:\n{results_text}\n\n"

        prompt += f"Available Operators: {operators}\n\n"
        prompt += "Based on the query and context, decide on the best action to take."

        return prompt
