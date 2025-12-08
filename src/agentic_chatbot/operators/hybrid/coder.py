"""Coder operator - hybrid LLM + MCP."""

import time
from typing import TYPE_CHECKING

from ..base import BaseOperator, OperatorType
from ..context import OperatorContext, OperatorResult
from ..registry import OperatorRegistry
from ...utils.llm import LLMClient

if TYPE_CHECKING:
    from ...mcp.session import MCPSession


@OperatorRegistry.register("coder")
class CoderOperator(BaseOperator):
    """Writes and executes code using LLM + sandbox."""

    name = "coder"
    description = "Write and run code in a sandbox"
    operator_type = OperatorType.HYBRID
    model = "sonnet"
    mcp_tools = ["run_python", "run_javascript"]
    context_requirements = ["query", "language"]

    def __init__(self):
        self._llm = LLMClient()

    async def execute(
        self,
        context: OperatorContext,
        mcp_session: "MCPSession | None" = None,
    ) -> OperatorResult:
        start_time = time.time()

        if not mcp_session:
            return OperatorResult(
                success=False,
                error="MCP session required for code execution",
                duration_ms=(time.time() - start_time) * 1000,
            )

        try:
            language = context.get_param("language", "python")

            code_response = await self._llm.complete(
                prompt=f"Write {language} code to accomplish: {context.query}\n\nOutput only the code, no explanations.",
                system="You are an expert programmer. Write clean, working code.",
                model=self.model or "sonnet",
            )

            code = code_response.content.strip()
            if code.startswith("```"):
                lines = code.split("\n")
                code = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            result = await mcp_session.call_tool(
                f"run_{language}",
                {"code": code},
            )

            return OperatorResult(
                success=result.success,
                output={
                    "code": code,
                    "execution_result": result.text,
                },
                error=result.error if not result.success else None,
                metadata={
                    "language": language,
                    "tokens_used": code_response.input_tokens + code_response.output_tokens,
                },
                duration_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return OperatorResult(
                success=False,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )
