"""Async LLM client wrapper for Anthropic Claude API."""

from dataclasses import dataclass
from typing import AsyncIterator

from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agentic_chatbot.config.settings import get_settings
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


# Model aliases for convenience
MODEL_ALIASES = {
    "haiku": "claude-3-haiku-20240307",
    "sonnet": "claude-3-5-sonnet-20241022",
    "opus": "claude-3-opus-20240229",
}


@dataclass
class LLMResponse:
    """Response from LLM call."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str


class LLMClient:
    """
    Async wrapper for Anthropic Claude API.

    Features:
    - Model aliases ("haiku", "sonnet", "opus")
    - Automatic retry with exponential backoff
    - Usage tracking (tokens, duration)
    - Streaming support
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize LLM client.

        Args:
            api_key: Anthropic API key (defaults to settings)
        """
        settings = get_settings()
        self._client = AsyncAnthropic(api_key=api_key or settings.anthropic_api_key)
        self._default_model = settings.default_model

    def _resolve_model(self, model: str) -> str:
        """Resolve model alias to full model ID."""
        return MODEL_ALIASES.get(model, model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    )
    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str = "sonnet",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """
        Call LLM and get complete response.

        Args:
            prompt: User prompt
            system: System prompt
            model: Model name or alias
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            LLMResponse with content and metadata
        """
        resolved_model = self._resolve_model(model)

        logger.debug(
            "Calling LLM",
            model=resolved_model,
            prompt_length=len(prompt),
            system_length=len(system),
        )

        messages = [{"role": "user", "content": prompt}]

        response = await self._client.messages.create(
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system if system else None,
            messages=messages,
        )

        content = ""
        if response.content and len(response.content) > 0:
            content = response.content[0].text

        logger.debug(
            "LLM response received",
            model=resolved_model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )

        return LLMResponse(
            content=content,
            model=resolved_model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )

    async def stream(
        self,
        prompt: str,
        system: str = "",
        model: str = "sonnet",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """
        Stream LLM response.

        Args:
            prompt: User prompt
            system: System prompt
            model: Model name or alias
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Yields:
            Response text chunks
        """
        resolved_model = self._resolve_model(model)

        logger.debug(
            "Starting LLM stream",
            model=resolved_model,
            prompt_length=len(prompt),
        )

        messages = [{"role": "user", "content": prompt}]

        async with self._client.messages.stream(
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system if system else None,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text


# Convenience function
async def call_llm(
    prompt: str,
    system: str = "",
    model: str = "sonnet",
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> str:
    """
    Convenience function to call LLM and get content.

    Args:
        prompt: User prompt
        system: System prompt
        model: Model name or alias
        max_tokens: Maximum tokens
        temperature: Sampling temperature

    Returns:
        Response content string
    """
    client = LLMClient()
    response = await client.complete(
        prompt=prompt,
        system=system,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.content
