"""Reflect node for quality evaluation and next action decision."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic_chatbot.config.prompts import REFLECT_SYSTEM_PROMPT, REFLECT_PROMPT
from agentic_chatbot.nodes.base import AsyncBaseNode
from agentic_chatbot.utils.structured_llm import StructuredLLMCaller
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class ReflectionResult(BaseModel):
    """Schema for reflection output."""

    quality_score: float = Field(
        ..., ge=0.0, le=1.0, description="Quality score from 0.0 to 1.0"
    )
    is_complete: bool = Field(..., description="Whether results are sufficient")
    issues: list[str] = Field(default_factory=list, description="Issues found")
    recommendation: Literal["satisfied", "need_more", "blocked"] = Field(
        ..., description="Recommended next action"
    )


class ReflectNode(AsyncBaseNode):
    """
    Evaluate results and decide next action.

    Type: Orchestration Node

    Reviews collected results and determines:
    - "satisfied": Results are good enough, proceed to response
    - "need_more": Need additional information, continue loop
    - "blocked": Cannot proceed, inform user of limitation
    """

    node_name = "reflect"
    description = "Evaluate results quality and completeness"

    async def prep_async(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Prepare reflection context."""
        observation = shared.get("observation", {})
        collected = observation.get("collected", [])
        stats = observation.get("stats", {})

        # Format results for prompt
        results_text = self._format_results(collected)

        return {
            "query": shared.get("user_query", ""),
            "results_text": results_text,
            "stats": stats,
            "iteration": shared.get("supervisor", {}).get("state", {}).iteration
            if hasattr(shared.get("supervisor", {}).get("state", {}), "iteration")
            else 0,
        }

    async def exec_async(self, prep_res: dict[str, Any]) -> ReflectionResult:
        """Execute reflection evaluation."""
        # Quick check: if no results, we need more
        if not prep_res["results_text"] or prep_res["results_text"] == "No results collected.":
            return ReflectionResult(
                quality_score=0.0,
                is_complete=False,
                issues=["No results collected"],
                recommendation="need_more",
            )

        # Use LLM for deeper evaluation
        caller = StructuredLLMCaller(max_retries=2)

        prompt = REFLECT_PROMPT.format(
            query=prep_res["query"],
            results=prep_res["results_text"],
        )

        try:
            result = await caller.call(
                prompt=prompt,
                response_model=ReflectionResult,
                system=REFLECT_SYSTEM_PROMPT,
                model="haiku",  # Fast evaluation
            )
            return result

        except Exception as e:
            logger.warning(f"Reflection LLM call failed: {e}")
            # Fallback: simple heuristic
            stats = prep_res["stats"]
            has_results = stats.get("total", 0) > 0
            all_successful = stats.get("successful", 0) == stats.get("total", 0)

            if has_results and all_successful:
                return ReflectionResult(
                    quality_score=0.7,
                    is_complete=True,
                    issues=[],
                    recommendation="satisfied",
                )
            elif has_results:
                return ReflectionResult(
                    quality_score=0.5,
                    is_complete=True,
                    issues=["Some results failed"],
                    recommendation="satisfied",  # Proceed with partial results
                )
            else:
                return ReflectionResult(
                    quality_score=0.0,
                    is_complete=False,
                    issues=["No results"],
                    recommendation="blocked",
                )

    async def post_async(
        self,
        shared: dict[str, Any],
        prep_res: dict[str, Any],
        exec_res: ReflectionResult,
    ) -> str:
        """Store reflection and return routing action."""
        shared.setdefault("reflection", {})
        shared["reflection"]["quality_score"] = exec_res.quality_score
        shared["reflection"]["is_complete"] = exec_res.is_complete
        shared["reflection"]["issues"] = exec_res.issues
        shared["reflection"]["recommendation"] = exec_res.recommendation

        logger.info(
            "Reflection complete",
            quality=exec_res.quality_score,
            recommendation=exec_res.recommendation,
        )

        return exec_res.recommendation

    def _format_results(self, collected: list[dict[str, Any]]) -> str:
        """Format collected results for prompt."""
        if not collected:
            return "No results collected."

        parts = []
        for item in collected:
            status = "✓" if item["success"] else "✗"
            parts.append(f"[{status}] {item['source']}:\n{item['content']}")

        return "\n\n".join(parts)
