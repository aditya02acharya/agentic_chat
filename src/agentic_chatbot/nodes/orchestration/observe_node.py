"""Observe node for collecting and compressing results."""

from typing import Any

from agentic_chatbot.nodes.base import AsyncBaseNode
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class ObserveNode(AsyncBaseNode):
    """
    Collect and compress results for supervisor review.

    Type: Orchestration Node

    Gathers outputs from tool calls or workflow execution
    and prepares them for the reflection step.
    """

    node_name = "observe"
    description = "Collect results for reflection"

    async def prep_async(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Gather results from execution."""
        results = shared.get("results", {})
        tool_outputs = results.get("tool_outputs", [])
        workflow_output = results.get("workflow_output")

        return {
            "tool_outputs": tool_outputs,
            "workflow_output": workflow_output,
            "query": shared.get("user_query", ""),
        }

    async def exec_async(self, prep_res: dict[str, Any]) -> dict[str, Any]:
        """Process and compress results."""
        tool_outputs = prep_res["tool_outputs"]
        workflow_output = prep_res["workflow_output"]

        # Collect all outputs
        collected = []

        for i, output in enumerate(tool_outputs):
            if hasattr(output, "text_output"):
                text = output.text_output[:2000]  # Truncate for context
            else:
                text = str(output)[:2000]
            collected.append({
                "source": f"tool_{i}",
                "content": text,
                "success": getattr(output, "success", True),
            })

        if workflow_output:
            if hasattr(workflow_output, "steps"):
                for step_id, result in workflow_output.steps.items():
                    collected.append({
                        "source": f"step_{step_id}",
                        "content": str(result.output)[:1000] if result.output else "",
                        "success": result.status.value == "completed",
                    })

        # Summary stats
        total = len(collected)
        successful = sum(1 for c in collected if c["success"])

        return {
            "collected": collected,
            "total_results": total,
            "successful_results": successful,
            "query": prep_res["query"],
        }

    async def post_async(
        self,
        shared: dict[str, Any],
        prep_res: dict[str, Any],
        exec_res: dict[str, Any],
    ) -> str | None:
        """Store observation for reflection."""
        shared.setdefault("observation", {})
        shared["observation"]["collected"] = exec_res["collected"]
        shared["observation"]["stats"] = {
            "total": exec_res["total_results"],
            "successful": exec_res["successful_results"],
        }

        logger.debug(
            "Observation complete",
            total=exec_res["total_results"],
            successful=exec_res["successful_results"],
        )

        return "default"
