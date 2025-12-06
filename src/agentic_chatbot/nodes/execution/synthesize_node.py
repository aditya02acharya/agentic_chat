"""Synthesize node for combining multiple sources."""

from typing import Any

from agentic_chatbot.nodes.base import AsyncBaseNode
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.operators.context import OperatorContext, OperatorResult
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class SynthesizeNode(AsyncBaseNode):
    """
    Combine multiple sources into coherent content.

    Type: Execution Node
    Uses: Synthesizer operator (Sonnet model)

    Takes results from multiple tool calls or workflow steps
    and synthesizes them into a unified response.
    """

    node_name = "synthesize"
    description = "Synthesize multiple sources into coherent content"

    async def prep_async(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Prepare synthesis context."""
        # Gather all sources
        results = shared.get("results", {})
        tool_outputs = results.get("tool_outputs", [])
        workflow_output = results.get("workflow_output")

        # Build sources dictionary
        sources = {}

        for i, output in enumerate(tool_outputs):
            key = f"tool_{i}"
            if hasattr(output, "output"):
                sources[key] = output.output
            else:
                sources[key] = output

        if workflow_output and hasattr(workflow_output, "steps"):
            for step_id, result in workflow_output.steps.items():
                if result.output:
                    sources[f"step_{step_id}"] = result.output

        return {
            "query": shared.get("user_query", ""),
            "sources": sources,
        }

    async def exec_async(self, prep_res: dict[str, Any]) -> OperatorResult:
        """Execute synthesis."""
        query = prep_res["query"]
        sources = prep_res["sources"]

        # Skip synthesis if only one source
        if len(sources) <= 1:
            if sources:
                single_key = list(sources.keys())[0]
                return OperatorResult.success_result(
                    output=sources[single_key],
                    metadata={"skipped_synthesis": True, "source": single_key},
                )
            return OperatorResult.error_result("No sources to synthesize")

        # Use synthesizer operator
        try:
            synthesizer = OperatorRegistry.create("synthesizer")
        except KeyError:
            # Fallback: simple concatenation
            logger.warning("Synthesizer operator not found, using fallback")
            combined = "\n\n".join(f"**{k}:**\n{v}" for k, v in sources.items())
            return OperatorResult.success_result(
                output=combined,
                metadata={"fallback": True},
            )

        context = OperatorContext(
            query=query,
            step_results=sources,
        )
        context.extra["sources"] = sources

        return await synthesizer.execute(context)

    async def post_async(
        self,
        shared: dict[str, Any],
        prep_res: dict[str, Any],
        exec_res: OperatorResult,
    ) -> str | None:
        """Store synthesized content."""
        shared.setdefault("results", {})
        shared["results"]["synthesis"] = exec_res.output

        logger.info(
            "Synthesis complete",
            success=exec_res.success,
            source_count=len(prep_res["sources"]),
        )

        return "default"
