"""Analyzer operator for examining data and providing insights."""

from typing import TYPE_CHECKING

from agentic_chatbot.config.prompts import ANALYZER_SYSTEM_PROMPT, ANALYZER_PROMPT
from agentic_chatbot.operators.base import BaseOperator, OperatorType
from agentic_chatbot.operators.context import OperatorContext, OperatorResult
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.utils.llm import LLMClient

if TYPE_CHECKING:
    from agentic_chatbot.mcp.session import MCPSession


@OperatorRegistry.register("analyzer")
class AnalyzerOperator(BaseOperator):
    """
    Analyzes data and provides insights.

    Type: PURE_LLM
    Model: Sonnet (needs strong analytical reasoning)

    Takes data and produces analysis with patterns, trends,
    and actionable insights.
    """

    name = "analyzer"
    description = "Analyzes data and provides insights"
    operator_type = OperatorType.PURE_LLM
    model = "sonnet"
    context_requirements = ["query", "data"]

    async def execute(
        self,
        context: OperatorContext,
        mcp_session: "MCPSession | None" = None,
    ) -> OperatorResult:
        """
        Execute data analysis.

        Args:
            context: Operator context with query and data to analyze
            mcp_session: Not used (pure LLM operator)

        Returns:
            OperatorResult with analysis
        """
        client = LLMClient()

        # Get data to analyze
        data = context.extra.get("data", "")
        if not data and context.step_results:
            import json

            data = json.dumps(context.step_results, indent=2)

        prompt = ANALYZER_PROMPT.format(
            query=context.query,
            data=data,
        )

        try:
            response = await client.complete(
                prompt=prompt,
                system=ANALYZER_SYSTEM_PROMPT,
                model=self.model or "sonnet",
            )

            return OperatorResult.success_result(
                output=response.content,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                metadata={"model": response.model},
            )

        except Exception as e:
            return OperatorResult.error_result(
                error=f"Analysis failed: {str(e)}",
            )
