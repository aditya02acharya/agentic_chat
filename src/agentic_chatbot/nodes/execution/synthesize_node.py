"""Synthesize node for combining multiple results."""

from typing import Any

from ..base import AsyncBaseNode
from ...operators.registry import OperatorRegistry
from ...operators.context import OperatorContext
from ...events.types import EventType
from ...utils.logging import get_logger

logger = get_logger(__name__)


class SynthesizeNode(AsyncBaseNode):
    """
    Synthesizes multiple results into a coherent response.

    Uses the synthesizer operator to combine information.
    """

    name = "synthesize"

    async def execute(self, shared: dict[str, Any]) -> str:
        previous_results = shared.get("previous_results", [])

        if len(previous_results) <= 1:
            if previous_results:
                shared["synthesized_content"] = str(previous_results[0])
            return "write"

        await self.emit_event(
            EventType.THINKING_UPDATE,
            {"phase": "synthesizing", "sources": len(previous_results)},
        )

        try:
            synthesizer = OperatorRegistry.get("synthesizer")
            context = OperatorContext(
                query=shared.get("query", self.ctx.user_query),
                previous_results=previous_results,
            )

            result = await synthesizer.execute(context)

            if result.success:
                shared["synthesized_content"] = result.output
            else:
                shared["synthesized_content"] = "\n\n".join(
                    str(r) for r in previous_results
                )

        except Exception as e:
            logger.error(f"Synthesis failed: {e}", exc_info=True)
            shared["synthesized_content"] = "\n\n".join(
                str(r) for r in previous_results
            )

        return "write"
