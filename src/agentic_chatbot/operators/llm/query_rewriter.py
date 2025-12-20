"""Query rewriter operator for optimizing search queries."""

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from agentic_chatbot.config.prompts import QUERY_REWRITER_SYSTEM_PROMPT, QUERY_REWRITER_PROMPT
from agentic_chatbot.operators.base import BaseOperator, OperatorType
from agentic_chatbot.operators.context import OperatorContext, OperatorResult
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.utils.structured_llm import StructuredLLMCaller

# New unified data model
from agentic_chatbot.data.execution import ExecutionInput, ExecutionOutput
from agentic_chatbot.data.content import JsonContent

if TYPE_CHECKING:
    from agentic_chatbot.mcp.session import MCPSession


class RewrittenQuery(BaseModel):
    """Schema for rewritten query output."""

    internal_query: str = Field(..., description="Query optimized for internal RAG search")
    external_query: str = Field(..., description="Query optimized for web search")
    keywords: list[str] = Field(default_factory=list, description="Key terms extracted")


@OperatorRegistry.register("query_rewriter")
class QueryRewriterOperator(BaseOperator):
    """
    Rewrites queries for better search results.

    Type: PURE_LLM (Fast Path)
    Model: Haiku (fast, cost-effective)

    Takes a user query and produces optimized versions for:
    - Internal knowledge base search (RAG)
    - External web search
    - Key term extraction
    """

    name = "query_rewriter"
    description = "Rewrites and expands queries for better search results"
    operator_type = OperatorType.PURE_LLM
    model = "haiku"
    context_requirements = ["query"]

    async def run(
        self,
        input: ExecutionInput,
        mcp_session: "MCPSession | None" = None,
    ) -> ExecutionOutput:
        """
        Execute query rewriting using unified data model.

        Args:
            input: ExecutionInput with query
            mcp_session: Not used (pure LLM operator)

        Returns:
            ExecutionOutput with rewritten queries as JsonContent
        """
        caller = StructuredLLMCaller(max_retries=2)

        prompt = QUERY_REWRITER_PROMPT.format(query=input.query)

        try:
            result = await caller.call(
                prompt=prompt,
                response_model=RewrittenQuery,
                system=QUERY_REWRITER_SYSTEM_PROMPT,
                model=self.model or "haiku",
            )

            output_data = {
                "internal_query": result.internal_query,
                "external_query": result.external_query,
                "keywords": result.keywords,
            }

            return ExecutionOutput.success(
                JsonContent(output_data),
                metadata={"original_query": input.query},
            )

        except Exception as e:
            return ExecutionOutput.error(f"Query rewriting failed: {str(e)}")

    async def execute(
        self,
        context: OperatorContext,
        mcp_session: "MCPSession | None" = None,
    ) -> OperatorResult:
        """
        Legacy interface - converts to new run() interface.

        DEPRECATED: Use run() with ExecutionInput instead.
        """
        exec_input = ExecutionInput(
            query=context.query,
            extra=context.extra,
            step_results={},
        )

        output = await self.run(exec_input, mcp_session)

        if output.success:
            # Extract JSON data for legacy output
            json_data = output.contents[0].data if output.contents else {}
            return OperatorResult.success_result(
                output=json_data,
                metadata=output.metadata,
            )
        else:
            return OperatorResult.error_result(error=output.error)
