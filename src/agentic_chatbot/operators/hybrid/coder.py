"""Coder operator that uses LLM for code generation and MCP for execution."""

from typing import TYPE_CHECKING

from agentic_chatbot.config.prompts import CODER_SYSTEM_PROMPT, CODER_PROMPT
from agentic_chatbot.core.exceptions import OperatorError
from agentic_chatbot.operators.base import BaseOperator, OperatorType
from agentic_chatbot.operators.context import OperatorContext, OperatorResult
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.utils.llm import LLMClient
from agentic_chatbot.mcp.models import ToolContent

if TYPE_CHECKING:
    from agentic_chatbot.mcp.session import MCPSession


@OperatorRegistry.register("coder")
class CoderOperator(BaseOperator):
    """
    Writes and executes code.

    Type: HYBRID (LLM + MCP)
    Model: Sonnet (needs strong coding ability)
    MCP Tools: run_python, run_javascript

    Uses LLM to generate code based on the task, then executes it
    via MCP sandbox tools.
    """

    name = "coder"
    description = "Write and run code"
    operator_type = OperatorType.HYBRID
    model = "sonnet"
    mcp_tools = ["run_python", "run_javascript"]
    context_requirements = ["query", "language", "tools.schema(run_python)"]

    async def execute(
        self,
        context: OperatorContext,
        mcp_session: "MCPSession | None" = None,
    ) -> OperatorResult:
        """
        Execute code generation and execution.

        Args:
            context: Operator context with query and language
            mcp_session: Required MCP session

        Returns:
            OperatorResult with code and execution result

        Raises:
            OperatorError: If MCP session not provided
        """
        if not mcp_session:
            raise OperatorError("MCP session required", operator_name=self.name)

        # Get language (default to Python)
        language = context.extra.get("language", "python").lower()
        if language not in ["python", "javascript"]:
            language = "python"

        client = LLMClient()

        # Step 1: LLM generates code
        prompt = CODER_PROMPT.format(
            task=context.query,
            language=language,
            context=context.extra.get("additional_context", "None"),
        )

        try:
            code_response = await client.complete(
                prompt=prompt,
                system=CODER_SYSTEM_PROMPT,
                model=self.model or "sonnet",
            )

            code = self._extract_code(code_response.content, language)

            # Step 2: MCP executes code
            tool_name = f"run_{language}"
            exec_result = await mcp_session.call_tool(
                tool_name,
                {"code": code},
            )

            # Combine results
            output = {
                "code": code,
                "language": language,
                "execution_result": exec_result.combined_text,
                "execution_status": exec_result.status.value,
            }

            # Build contents with code and result
            contents = [
                ToolContent.markdown(f"```{language}\n{code}\n```"),
            ]
            if exec_result.contents:
                contents.extend(exec_result.contents)

            return OperatorResult.success_result(
                output=output,
                contents=contents,
                input_tokens=code_response.input_tokens,
                output_tokens=code_response.output_tokens,
                metadata={
                    "model": code_response.model,
                    "execution_duration_ms": exec_result.duration_ms,
                },
            )

        except Exception as e:
            return OperatorResult.error_result(
                error=f"Code execution failed: {str(e)}",
            )

    def _extract_code(self, response: str, language: str) -> str:
        """Extract code from LLM response."""
        # Look for code blocks
        markers = [f"```{language}", "```"]
        for marker in markers:
            if marker in response:
                start = response.find(marker)
                # Skip the marker line
                start = response.find("\n", start) + 1
                end = response.find("```", start)
                if end > start:
                    return response[start:end].strip()

        # If no code block found, assume entire response is code
        # (after removing any explanation lines)
        lines = response.strip().split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            # Skip explanation lines
            if not in_code and (
                line.startswith("#") or line.startswith("//") or not line.strip()
            ):
                continue
            in_code = True
            code_lines.append(line)

        return "\n".join(code_lines)
