"""Base interface for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Any

from agentic_chatbot.config.models import (
    ModelRegistry,
    ThinkingConfig,
    TokenUsage,
)


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""

    content: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    stop_reason: str = ""
    provider: str = "unknown"

    # Thinking mode outputs
    thinking_content: str = ""  # The model's thinking process (if enabled)

    # Legacy fields for backwards compatibility
    @property
    def input_tokens(self) -> int:
        return self.usage.input_tokens

    @property
    def output_tokens(self) -> int:
        return self.usage.output_tokens


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers (Anthropic, Bedrock, OpenAI, etc.) implement this interface
    to ensure consistent behavior across the application.

    Supports:
    - Model resolution via ModelRegistry
    - Extended thinking mode
    - Comprehensive token tracking
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'anthropic', 'bedrock')."""
        ...

    def resolve_model(self, model: str) -> str:
        """
        Resolve model alias to provider-specific model ID.

        Args:
            model: Model ID or alias (e.g., "sonnet", "haiku", "thinking")

        Returns:
            Provider-specific model ID
        """
        config = ModelRegistry.get(model)
        if config:
            return config.get_provider_id(self.provider_name)
        # Fall back to direct model ID
        return model

    def get_model_config(self, model: str):
        """Get model configuration."""
        from agentic_chatbot.config.models import get_model

        try:
            return get_model(model)
        except ValueError:
            return None

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str = "sonnet",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        thinking: ThinkingConfig | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Call LLM and get complete response.

        Args:
            prompt: User prompt
            system: System prompt
            model: Model name or alias
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            thinking: Extended thinking configuration
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with content, usage, and optional thinking
        """
        ...

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        system: str = "",
        model: str = "sonnet",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        thinking: ThinkingConfig | None = None,
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
            thinking: Extended thinking configuration
            **kwargs: Additional provider-specific parameters

        Yields:
            Response text chunks
        """
        ...
        # Make this a generator
        if False:
            yield ""
