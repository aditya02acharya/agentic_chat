"""Base interface for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    provider: str = "unknown"


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers (Anthropic, Bedrock, OpenAI, etc.) implement this interface
    to ensure consistent behavior across the application.
    """

    # Model aliases mapping - override in subclasses if different
    MODEL_ALIASES: dict[str, str] = {
        "haiku": "claude-3-haiku-20240307",
        "sonnet": "claude-3-5-sonnet-20241022",
        "opus": "claude-3-opus-20240229",
    }

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'anthropic', 'bedrock')."""
        ...

    def resolve_model(self, model: str) -> str:
        """Resolve model alias to full model ID."""
        return self.MODEL_ALIASES.get(model, model)

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...
        # Make this a generator
        if False:
            yield ""
