"""Async LLM client wrapper with multi-provider support.

Supports:
- Anthropic direct API (default)
- AWS Bedrock

The provider is selected based on the LLM_PROVIDER environment variable.
"""

from dataclasses import dataclass
from typing import AsyncIterator

from agentic_chatbot.config.settings import get_settings
from agentic_chatbot.utils.logging import get_logger
from agentic_chatbot.utils.providers import (
    BaseLLMProvider,
    LLMResponse,
    create_provider,
)


logger = get_logger(__name__)


# Re-export LLMResponse for backwards compatibility
# (it's now defined in providers.base)
__all__ = ["LLMClient", "LLMResponse", "call_llm"]


class LLMClient:
    """
    Async LLM client with multi-provider support.

    Features:
    - Multiple providers (Anthropic direct, AWS Bedrock)
    - Model aliases ("haiku", "sonnet", "opus")
    - Automatic retry with exponential backoff
    - Usage tracking (tokens)
    - Streaming support

    The provider is automatically selected based on settings,
    or can be explicitly specified.

    Example:
        # Use default provider from settings
        client = LLMClient()

        # Explicit Anthropic
        client = LLMClient(provider="anthropic", api_key="sk-...")

        # Explicit Bedrock
        client = LLMClient(provider="bedrock", region_name="us-west-2")
    """

    def __init__(
        self,
        provider: str | None = None,
        **provider_kwargs,
    ):
        """
        Initialize LLM client.

        Args:
            provider: Provider name ("anthropic" or "bedrock"). Uses settings if not specified.
            **provider_kwargs: Provider-specific arguments (api_key, region_name, etc.)
        """
        settings = get_settings()
        self._provider: BaseLLMProvider = create_provider(provider, **provider_kwargs)
        self._default_model = settings.default_model

        logger.info(
            "LLM client initialized",
            provider=self._provider.provider_name,
        )

    @property
    def provider_name(self) -> str:
        """Get the name of the current provider."""
        return self._provider.provider_name

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
            model: Model name or alias ("haiku", "sonnet", "opus")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            LLMResponse with content and metadata
        """
        return await self._provider.complete(
            prompt=prompt,
            system=system,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
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
        async for chunk in self._provider.stream(
            prompt=prompt,
            system=system,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            yield chunk

    # Backwards compatibility - expose generate as alias for complete
    async def generate(
        self,
        prompt: str,
        system: str = "",
        model: str = "sonnet",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate response from LLM (backwards compatibility).

        Args:
            prompt: User prompt
            system: System prompt
            model: Model name or alias
            max_tokens: Maximum tokens
            temperature: Sampling temperature

        Returns:
            Response content string
        """
        response = await self.complete(
            prompt=prompt,
            system=system,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.content


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
