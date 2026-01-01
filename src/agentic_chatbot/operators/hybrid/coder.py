"""Coder operator that uses LLM for code generation and MCP for execution."""

import re
from typing import TYPE_CHECKING

from agentic_chatbot.config.prompts import CODER_SYSTEM_PROMPT, CODER_PROMPT
from agentic_chatbot.core.exceptions import OperatorError
from agentic_chatbot.operators.base import BaseOperator, OperatorType
from agentic_chatbot.operators.context import OperatorContext, OperatorResult
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.utils.llm import LLMClient
from agentic_chatbot.mcp.models import ToolContent
from agentic_chatbot.utils.logging import get_logger

# New unified data model
from agentic_chatbot.data.execution import ExecutionInput, ExecutionOutput
from agentic_chatbot.data.content import TextContent, JsonContent

if TYPE_CHECKING:
    from agentic_chatbot.mcp.session import MCPSession


logger = get_logger(__name__)


# Keywords that indicate complex coding tasks requiring extended thinking
COMPLEX_TASK_KEYWORDS = [
    "algorithm", "optimize", "performance", "recursive", "dynamic programming",
    "tree", "graph", "traversal", "binary search", "sorting", "data structure",
    "concurrent", "async", "parallel", "thread", "multiprocess",
    "design pattern", "architecture", "refactor", "debug", "fix",
    "test", "unit test", "integration", "security", "vulnerability",
    "memory", "efficient", "complexity", "big o", "cache",
]


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

    async def run(
        self,
        input: ExecutionInput,
        mcp_session: "MCPSession | None" = None,
    ) -> ExecutionOutput:
        """
        Execute code generation and execution using unified data model.

        Uses extended thinking mode for complex coding tasks that require:
        - Algorithm design
        - Performance optimization
        - Debugging/fixing issues
        - Security considerations
        - Complex data structures

        Args:
            input: ExecutionInput with query and language
            mcp_session: Required MCP session

        Returns:
            ExecutionOutput with code and execution result

        Raises:
            OperatorError: If MCP session not provided
        """
        if not mcp_session:
            raise OperatorError("MCP session required", operator_name=self.name)

        # Get language (default to Python)
        language = input.get("language", "python").lower()
        if language not in ["python", "javascript"]:
            language = "python"

        # Determine if task requires extended thinking
        use_thinking = self._should_use_thinking(input.query)

        client = LLMClient()

        # Step 1: LLM generates code
        prompt = CODER_PROMPT.format(
            task=input.query,
            language=language,
            context=input.get("additional_context", "None"),
        )

        try:
            # Use thinking mode for complex tasks
            if use_thinking:
                logger.info(
                    "Using extended thinking for complex coding task",
                    task_preview=input.query[:100],
                )
                code_response = await client.complete(
                    prompt=prompt,
                    system=CODER_SYSTEM_PROMPT,
                    model="thinking",
                    enable_thinking=True,
                    thinking_budget=15000,
                )
            else:
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

            # Combine results as ContentBlocks
            output_data = {
                "code": code,
                "language": language,
                "execution_result": exec_result.combined_text,
                "execution_status": exec_result.status.value,
            }

            # Build contents with code and result
            contents = [
                TextContent.code(code, language=language),
            ]
            if exec_result.combined_text:
                contents.append(TextContent.plain(exec_result.combined_text))

            return ExecutionOutput(
                contents=contents,
                input_tokens=code_response.usage.input_tokens,
                output_tokens=code_response.usage.output_tokens,
                thinking_tokens=code_response.usage.thinking_tokens,
                metadata={
                    "model": code_response.model,
                    "execution_duration_ms": exec_result.duration_ms,
                    "used_thinking": use_thinking,
                    "output_data": output_data,
                },
            )

        except Exception as e:
            return ExecutionOutput.error(f"Code execution failed: {str(e)}")

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

    def _should_use_thinking(self, query: str) -> bool:
        """
        Determine if the coding task requires extended thinking.

        Complex tasks that benefit from thinking mode:
        - Algorithm design and optimization
        - Debugging and fixing issues
        - Performance optimization
        - Security considerations
        - Complex data structures

        Args:
            query: The coding task description

        Returns:
            True if extended thinking should be used
        """
        query_lower = query.lower()

        # Check for complex task keywords
        for keyword in COMPLEX_TASK_KEYWORDS:
            if keyword in query_lower:
                return True

        # Check for long/complex queries (likely more complex tasks)
        if len(query) > 300:
            return True

        # Check for multiple requirements (indicated by bullet points or numbers)
        if re.search(r"(\d+\.|[-*])\s+\w+", query):
            return True

        return False

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
            # Extract output data from metadata
            output_data = output.metadata.get("output_data", {})
            return OperatorResult.success_result(
                output=output_data,
                input_tokens=output.input_tokens,
                output_tokens=output.output_tokens,
                metadata=output.metadata,
            )
        else:
            return OperatorResult.error_result(error=output.error)
