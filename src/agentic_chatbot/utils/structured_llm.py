"""Structured LLM output with Pydantic validation, retry, and thinking support."""

import json
from dataclasses import dataclass, field
from typing import TypeVar, Type, Any, Generic

from pydantic import BaseModel, ValidationError

from agentic_chatbot.config.models import TokenUsage, get_thinking_config
from agentic_chatbot.utils.llm import LLMClient, LLMResponse
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """Raised when LLM output cannot be validated after retries."""

    def __init__(self, message: str, attempts: list[dict[str, Any]]):
        super().__init__(message)
        self.attempts = attempts  # History of failed attempts


@dataclass
class StructuredResult(Generic[T]):
    """Result from structured LLM call including token usage."""

    data: T
    usage: TokenUsage = field(default_factory=TokenUsage)
    thinking_content: str = ""  # Model's thinking process (if enabled)
    model: str = ""
    attempts: int = 1


class StructuredLLMCaller:
    """
    Calls LLM expecting structured JSON output with validation and retry.

    Features:
    - Pydantic schema validation
    - Error-feedback retry (tells LLM what went wrong)
    - Extended thinking mode support
    - Token usage tracking across retries
    - Configurable max retries
    - Preserves original inputs on retry

    Design Pattern: Retry with Feedback
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        max_retries: int = 3,
    ):
        """
        Initialize structured LLM caller.

        Args:
            llm_client: LLM client instance (creates default if not provided)
            max_retries: Maximum number of retry attempts
        """
        self.llm_client = llm_client or LLMClient()
        self.max_retries = max_retries

    async def call(
        self,
        prompt: str,
        response_model: Type[T],
        system: str | None = None,
        model: str = "sonnet",
        enable_thinking: bool = False,
        thinking_budget: int | None = None,
    ) -> T:
        """
        Call LLM and parse response into Pydantic model.

        Args:
            prompt: User prompt requesting structured output
            response_model: Pydantic model class for validation
            system: System prompt (will append schema instructions)
            model: LLM model to use
            enable_thinking: Enable extended thinking mode
            thinking_budget: Custom thinking token budget

        Returns:
            Validated Pydantic model instance

        Raises:
            StructuredOutputError: After max_retries failures
        """
        result = await self.call_with_usage(
            prompt=prompt,
            response_model=response_model,
            system=system,
            model=model,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
        )
        return result.data

    async def call_with_usage(
        self,
        prompt: str,
        response_model: Type[T],
        system: str | None = None,
        model: str = "sonnet",
        enable_thinking: bool = False,
        thinking_budget: int | None = None,
    ) -> StructuredResult[T]:
        """
        Call LLM and parse response, returning result with token usage.

        Args:
            prompt: User prompt requesting structured output
            response_model: Pydantic model class for validation
            system: System prompt (will append schema instructions)
            model: LLM model to use
            enable_thinking: Enable extended thinking mode
            thinking_budget: Custom thinking token budget

        Returns:
            StructuredResult with validated data and usage info

        Raises:
            StructuredOutputError: After max_retries failures
        """
        # Build schema instruction
        schema_json = response_model.model_json_schema()
        schema_instruction = self._build_schema_instruction(schema_json)

        full_system = f"{system or ''}\n\n{schema_instruction}".strip()
        attempts: list[dict[str, Any]] = []
        current_prompt = prompt

        # Track total usage across retries
        total_usage = TokenUsage()
        thinking_content = ""
        final_model = model

        for attempt in range(self.max_retries):
            # Call LLM
            response = await self.llm_client.complete(
                prompt=current_prompt,
                system=full_system,
                model=model,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
            )

            # Accumulate token usage
            total_usage = total_usage + response.usage
            if response.thinking_content:
                thinking_content = response.thinking_content
            final_model = response.model

            raw_output = response.content

            # Try to parse JSON
            try:
                # Extract JSON from response (handle markdown code blocks)
                json_str = self._extract_json(raw_output)
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON: {e.msg} at position {e.pos}"
                logger.warning(
                    "JSON parse error on attempt",
                    attempt=attempt + 1,
                    error=error_msg,
                )
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "raw_output": raw_output[:500],
                        "error_type": "json_parse",
                        "error": error_msg,
                    }
                )
                current_prompt = self._build_retry_prompt(
                    original_prompt=prompt,
                    error_type="JSON Parse Error",
                    error_details=error_msg,
                    raw_output=raw_output,
                )
                continue

            # Try to validate against Pydantic schema
            try:
                result = response_model.model_validate(data)
                logger.debug(
                    "Successfully parsed structured output",
                    model=response_model.__name__,
                    attempts=attempt + 1,
                )
                return StructuredResult(
                    data=result,
                    usage=total_usage,
                    thinking_content=thinking_content,
                    model=final_model,
                    attempts=attempt + 1,
                )
            except ValidationError as e:
                error_msg = self._format_validation_errors(e)
                logger.warning(
                    "Schema validation error on attempt",
                    attempt=attempt + 1,
                    error=error_msg,
                )
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "raw_output": raw_output[:500],
                        "parsed_json": data,
                        "error_type": "schema_validation",
                        "error": error_msg,
                    }
                )
                current_prompt = self._build_retry_prompt(
                    original_prompt=prompt,
                    error_type="Schema Validation Error",
                    error_details=error_msg,
                    raw_output=raw_output,
                )
                continue

        # All retries exhausted
        logger.error(
            "Failed to get valid structured output after retries",
            max_retries=self.max_retries,
            model=response_model.__name__,
        )
        raise StructuredOutputError(
            f"Failed to get valid structured output after {self.max_retries} attempts",
            attempts=attempts,
        )

    def _build_schema_instruction(self, schema: dict[str, Any]) -> str:
        """Build instruction telling LLM the expected schema."""
        return f"""You MUST respond with valid JSON matching this schema:

```json
{json.dumps(schema, indent=2)}
```

Rules:
1. Output ONLY valid JSON, no explanations before or after
2. All required fields must be present
3. Field types must match the schema exactly
4. Use null for optional fields if not applicable"""

    def _build_retry_prompt(
        self,
        original_prompt: str,
        error_type: str,
        error_details: str,
        raw_output: str,
    ) -> str:
        """Build prompt for retry with error feedback."""
        return f"""{original_prompt}

---
PREVIOUS ATTEMPT FAILED - Please fix and try again.

Error Type: {error_type}
Error Details: {error_details}

Your previous output was:
```
{raw_output[:1000]}
```

Please provide a corrected JSON response that fixes these issues."""

    def _extract_json(self, text: str) -> str:
        """Extract JSON from response, handling markdown code blocks."""
        text = text.strip()

        # Try to extract from markdown code block
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        if "```" in text:
            start = text.find("```") + 3
            # Skip optional language identifier
            newline = text.find("\n", start)
            if newline != -1 and newline - start < 20:
                start = newline + 1
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # Try to find JSON object/array directly
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            if start != -1:
                # Find matching closing bracket
                depth = 0
                in_string = False
                escape = False
                for i, char in enumerate(text[start:], start):
                    if escape:
                        escape = False
                        continue
                    if char == "\\":
                        escape = True
                        continue
                    if char == '"':
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if char == start_char:
                        depth += 1
                    elif char == end_char:
                        depth -= 1
                        if depth == 0:
                            return text[start : i + 1]

        return text

    def _format_validation_errors(self, error: ValidationError) -> str:
        """Format Pydantic validation errors for LLM feedback."""
        errors = []
        for e in error.errors():
            loc = " -> ".join(str(x) for x in e["loc"])
            errors.append(f"- Field '{loc}': {e['msg']}")
        return "\n".join(errors)
