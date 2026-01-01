"""Async LLM client wrapper with multi-provider support.

Supports:
- Anthropic direct API (default)
- AWS Bedrock
- Extended thinking mode
- Comprehensive token tracking
- Model resolution via ModelRegistry

The provider is selected based on the LLM_PROVIDER environment variable.
"""

from typing import AsyncIterator, Any

from agentic_chatbot.config.models import (
    ThinkingConfig,
    TokenUsage,
    ModelRegistry,
    get_model,
    get_thinking_config,
)
from agentic_chatbot.config.settings import get_settings
from agentic_chatbot.utils.logging import get_logger
from agentic_chatbot.utils.providers import (
    BaseLLMProvider,
    LLMResponse,
    create_provider,
)


logger = get_logger(__name__)


# Re-export for backwards compatibility
__all__ = [
    "LLMClient",
    "LLMResponse",
    "TokenUsage",
    "ThinkingConfig",
    "call_llm",
    "call_llm_with_thinking",
]


class LLMClient:
    """
    Async LLM client with multi-provider support.

    Features:
    - Multiple providers (Anthropic direct, AWS Bedrock)
    - Model resolution via ModelRegistry (aliases: "haiku", "sonnet", "opus", "thinking")
    - Extended thinking mode support
    - Comprehensive token tracking
    - Automatic retry with exponential backoff
    - Streaming support

    The provider is automatically selected based on settings,
    or can be explicitly specified.

    Example:
        # Use default provider from settings
        client = LLMClient()

        # Simple completion
        response = await client.complete("What is Python?")

        # With thinking mode
        response = await client.complete(
            "Solve this complex problem...",
            model="thinking",  # Uses thinking-enabled model
            enable_thinking=True,
            thinking_budget=20000,
        )
        print(response.thinking_content)  # The model's reasoning
        print(response.content)  # The final answer
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
        enable_thinking: bool = False,
        thinking_budget: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Call LLM and get complete response.

        Args:
            prompt: User prompt
            system: System prompt
            model: Model name or alias ("haiku", "sonnet", "opus", "thinking", or full model ID)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (ignored if thinking enabled)
            enable_thinking: Enable extended thinking mode
            thinking_budget: Custom thinking token budget (uses model default if not specified)
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with content, usage, and optional thinking output
        """
        # Get thinking configuration
        thinking = None
        if enable_thinking:
            thinking = get_thinking_config(model, True, thinking_budget)

        return await self._provider.complete(
            prompt=prompt,
            system=system,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=thinking,
            **kwargs,
        )

    async def stream(
        self,
        prompt: str,
        system: str = "",
        model: str = "sonnet",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        thinking_budget: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream LLM response.

        Args:
            prompt: User prompt
            system: System prompt
            model: Model name or alias
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            enable_thinking: Enable extended thinking mode
            thinking_budget: Custom thinking token budget
            **kwargs: Additional provider-specific parameters

        Yields:
            Response text chunks

        Note:
            When thinking is enabled, the thinking content is not streamed.
            Use complete() instead if you need the thinking output.
        """
        thinking = None
        if enable_thinking:
            thinking = get_thinking_config(model, True, thinking_budget)

        async for chunk in self._provider.stream(
            prompt=prompt,
            system=system,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=thinking,
            **kwargs,
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

    def get_model_info(self, model: str) -> dict[str, Any]:
        """
        Get information about a model.

        Args:
            model: Model ID or alias

        Returns:
            Dictionary with model info (supports_thinking, category, etc.)
        """
        try:
            config = get_model(model)
            return {
                "id": config.id,
                "name": config.name,
                "category": config.category.value,
                "supports_thinking": config.supports_thinking,
                "max_output_tokens": config.max_output_tokens,
                "context_window": config.context_window,
            }
        except ValueError:
            return {"id": model, "name": model, "supports_thinking": False}

    @staticmethod
    def list_models() -> list[dict[str, Any]]:
        """List all available models."""
        return [
            {
                "id": m.id,
                "name": m.name,
                "aliases": m.aliases,
                "category": m.category.value,
                "supports_thinking": m.supports_thinking,
            }
            for m in ModelRegistry.list_all()
        ]


# Convenience functions


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


async def call_llm_with_thinking(
    prompt: str,
    system: str = "",
    model: str = "thinking",
    max_tokens: int = 8192,
    thinking_budget: int = 10000,
) -> LLMResponse:
    """
    Convenience function to call LLM with extended thinking.

    Args:
        prompt: User prompt
        system: System prompt
        model: Model name or alias (default: "thinking")
        max_tokens: Maximum output tokens
        thinking_budget: Thinking token budget

    Returns:
        LLMResponse with content and thinking_content
    """
    client = LLMClient()
    return await client.complete(
        prompt=prompt,
        system=system,
        model=model,
        max_tokens=max_tokens,
        enable_thinking=True,
        thinking_budget=thinking_budget,
    )
