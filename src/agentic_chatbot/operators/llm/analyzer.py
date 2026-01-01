"""Analyzer operator for examining data and providing insights."""

from typing import TYPE_CHECKING

from agentic_chatbot.config.prompts import ANALYZER_SYSTEM_PROMPT, ANALYZER_PROMPT
from agentic_chatbot.operators.base import BaseOperator, OperatorType
from agentic_chatbot.operators.context import OperatorContext, OperatorResult
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.utils.llm import LLMClient

# New unified data model
from agentic_chatbot.data.execution import ExecutionInput, ExecutionOutput
from agentic_chatbot.data.content import TextContent

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

    async def run(
        self,
        input: ExecutionInput,
        mcp_session: "MCPSession | None" = None,
    ) -> ExecutionOutput:
        """
        Execute data analysis using unified data model.

        Args:
            input: ExecutionInput with query and data to analyze
            mcp_session: Not used (pure LLM operator)

        Returns:
            ExecutionOutput with analysis as ContentBlock
        """
        client = LLMClient()

        # Get data to analyze
        data = input.get("data", "")
        if not data and input.step_results:
            import json
            # Convert step results to JSON for analysis
            data = json.dumps(
                {k: v.text for k, v in input.step_results.items()},
                indent=2,
            )

        prompt = ANALYZER_PROMPT.format(
            query=input.query,
            data=data,
        )

        try:
            response = await client.complete(
                prompt=prompt,
                system=ANALYZER_SYSTEM_PROMPT,
                model=self.model or "sonnet",
            )

            return ExecutionOutput.success(
                TextContent.markdown(response.content),
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                metadata={"model": response.model},
            )

        except Exception as e:
            return ExecutionOutput.error(f"Analysis failed: {str(e)}")

    async def execute(
        self,
        context: OperatorContext,
        mcp_session: "MCPSession | None" = None,
    ) -> OperatorResult:
        """
        Legacy interface - converts to new run() interface.

        DEPRECATED: Use run() with ExecutionInput instead.
        """
        # Convert context to ExecutionInput and call run()
        exec_input = ExecutionInput(
            query=context.query,
            extra=context.extra,
            step_results={},  # Legacy context doesn't have typed step results
        )

        output = await self.run(exec_input, mcp_session)

        # Convert ExecutionOutput back to OperatorResult
        if output.success:
            return OperatorResult.success_result(
                output=output.text,
                input_tokens=output.input_tokens,
                output_tokens=output.output_tokens,
                metadata=output.metadata,
            )
        else:
            return OperatorResult.error_result(error=output.error)
