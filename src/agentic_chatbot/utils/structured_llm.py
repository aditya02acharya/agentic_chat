"""Structured LLM output with Pydantic validation and retry."""

import json
from typing import TypeVar, Type

from pydantic import BaseModel, ValidationError

from .llm import LLMClient
from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """Raised when LLM output cannot be validated after retries."""

    def __init__(self, message: str, attempts: list[dict]):
        super().__init__(message)
        self.attempts = attempts


class StructuredLLMCaller:
    """
    Calls LLM expecting structured JSON output with validation and retry.

    Features:
    - Pydantic schema validation
    - Error-feedback retry (tells LLM what went wrong)
    - Configurable max retries
    - Preserves original inputs on retry
    """

    def __init__(self, llm_client: LLMClient, max_retries: int = 3):
        self.llm_client = llm_client
        self.max_retries = max_retries

    async def call(
        self,
        prompt: str,
        response_model: Type[T],
        system: str | None = None,
        model: str = "sonnet",
    ) -> T:
        """
        Call LLM and parse response into Pydantic model.

        Args:
            prompt: User prompt requesting structured output
            response_model: Pydantic model class for validation
            system: System prompt (will append schema instructions)
            model: LLM model to use

        Returns:
            Validated Pydantic model instance

        Raises:
            StructuredOutputError: After max_retries failures
        """
        schema_json = response_model.model_json_schema()
        schema_instruction = self._build_schema_instruction(schema_json)

        full_system = f"{system or ''}\n\n{schema_instruction}".strip()
        attempts: list[dict] = []
        current_prompt = prompt

        for attempt in range(self.max_retries):
            response = await self.llm_client.complete(
                prompt=current_prompt,
                system=full_system,
                model=model,
            )

            raw_output = response.content

            try:
                json_str = self._extract_json(raw_output)
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON: {e.msg} at position {e.pos}"
                attempts.append({
                    "attempt": attempt + 1,
                    "raw_output": raw_output[:500],
                    "error_type": "json_parse",
                    "error": error_msg,
                })
                current_prompt = self._build_retry_prompt(
                    original_prompt=prompt,
                    error_type="JSON Parse Error",
                    error_details=error_msg,
                    raw_output=raw_output,
                )
                continue

            try:
                result = response_model.model_validate(data)
                return result
            except ValidationError as e:
                error_msg = self._format_validation_errors(e)
                attempts.append({
                    "attempt": attempt + 1,
                    "raw_output": raw_output[:500],
                    "parsed_json": data,
                    "error_type": "schema_validation",
                    "error": error_msg,
                })
                current_prompt = self._build_retry_prompt(
                    original_prompt=prompt,
                    error_type="Schema Validation Error",
                    error_details=error_msg,
                    raw_output=raw_output,
                )
                continue

        raise StructuredOutputError(
            f"Failed to get valid structured output after {self.max_retries} attempts",
            attempts=attempts,
        )

    def _build_schema_instruction(self, schema: dict) -> str:
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

        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            if start != -1:
                depth = 0
                for i, char in enumerate(text[start:], start):
                    if char == start_char:
                        depth += 1
                    elif char == end_char:
                        depth -= 1
                        if depth == 0:
                            return text[start:i + 1]

        return text

    def _format_validation_errors(self, error: ValidationError) -> str:
        """Format Pydantic validation errors for LLM feedback."""
        errors = []
        for e in error.errors():
            loc = " -> ".join(str(x) for x in e["loc"])
            errors.append(f"- Field '{loc}': {e['msg']}")
        return "\n".join(errors)
