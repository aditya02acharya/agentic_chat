"""Anthropic direct API provider."""

from typing import AsyncIterator

from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agentic_chatbot.utils.providers.base import BaseLLMProvider, LLMResponse
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """
    Direct Anthropic API provider.

    Uses the official Anthropic Python SDK to call Claude models directly.

    Features:
    - Model aliases ("haiku", "sonnet", "opus")
    - Automatic retry with exponential backoff
    - Usage tracking (tokens)
    - Streaming support
    """

    def __init__(self, api_key: str):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
        """
        self._client = AsyncAnthropic(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

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
        """Call Anthropic API and get complete response."""
        resolved_model = self.resolve_model(model)

        logger.debug(
            "Calling Anthropic API",
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
            "Anthropic response received",
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
            provider=self.provider_name,
        )

    async def stream(
        self,
        prompt: str,
        system: str = "",
        model: str = "sonnet",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Stream response from Anthropic API."""
        resolved_model = self.resolve_model(model)

        logger.debug(
            "Starting Anthropic stream",
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
