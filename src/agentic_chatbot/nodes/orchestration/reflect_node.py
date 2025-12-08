"""Reflect node for quality evaluation."""

from typing import Any

from ..base import AsyncBaseNode
from ...core.supervisor import ReflectionResult
from ...config.prompts import REFLECT_SYSTEM_PROMPT
from ...events.types import EventType
from ...utils.llm import LLMClient
from ...utils.structured_llm import StructuredLLMCaller, StructuredOutputError
from ...utils.logging import get_logger

logger = get_logger(__name__)


class ReflectNode(AsyncBaseNode):
    """
    Evaluates quality and decides next steps.

    Determines if results are satisfactory or if more work is needed.
    """

    name = "reflect"

    def __init__(self, ctx):
        super().__init__(ctx)
        self._llm = LLMClient()
        self._structured_llm = StructuredLLMCaller(self._llm)

    async def execute(self, shared: dict[str, Any]) -> str:
        query = shared.get("query", self.ctx.user_query)
        previous_results = shared.get("previous_results", [])
        iteration = shared.get("iteration", 0)
        max_iterations = shared.get("max_iterations", 5)

        if iteration >= max_iterations:
            shared["reflection"] = ReflectionResult(
                quality_score=0.5,
                is_complete=True,
                issues=["Maximum iterations reached"],
                recommendation="satisfied",
                reasoning="Forced completion due to iteration limit",
            )
            return "satisfied"

        if not previous_results:
            shared["reflection"] = ReflectionResult(
                quality_score=0.0,
                is_complete=False,
                issues=["No results to evaluate"],
                recommendation="need_more",
                reasoning="No data collected yet",
            )
            return "need_more"

        results_text = "\n\n".join(str(r)[:500] for r in previous_results[-3:])
        prompt = f"""User Query: {query}

Results Collected:
{results_text}

Evaluate whether these results adequately answer the user's query."""

        try:
            reflection = await self._structured_llm.call(
                prompt=prompt,
                response_model=ReflectionResult,
                system=REFLECT_SYSTEM_PROMPT,
                model="haiku",
            )
            shared["reflection"] = reflection

            await self.emit_event(
                EventType.THINKING_UPDATE,
                {
                    "phase": "reflection",
                    "quality_score": reflection.quality_score,
                    "recommendation": reflection.recommendation,
                },
            )

            return reflection.recommendation

        except StructuredOutputError:
            if len(previous_results) >= 2:
                return "satisfied"
            return "need_more"
