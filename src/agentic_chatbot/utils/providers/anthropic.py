"""Anthropic direct API provider with extended thinking support.

Resilience patterns applied:
- Retry with exponential backoff for transient failures
- Circuit breaker to prevent cascade failures to overloaded API
- Timeout to bound operation duration
"""

from typing import AsyncIterator, Any

from anthropic import AsyncAnthropic

from agentic_chatbot.config.models import ThinkingConfig, TokenUsage
from agentic_chatbot.core.resilience import (
    llm_retry,
    llm_circuit_breaker,
    llm_timeout,
    wrap_anthropic_errors,
    TransientError,
    RateLimitError,
    BreakerOpen,
)
from agentic_chatbot.utils.providers.base import BaseLLMProvider, LLMResponse
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """
    Direct Anthropic API provider.

    Uses the official Anthropic Python SDK to call Claude models directly.

    Features:
    - Model aliases via ModelRegistry
    - Extended thinking mode support
    - Comprehensive token tracking
    - Automatic retry with exponential backoff
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

    @llm_retry
    @llm_circuit_breaker
    @llm_timeout
    @wrap_anthropic_errors
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
        Call Anthropic API with optional extended thinking.

        Args:
            prompt: User prompt
            system: System prompt
            model: Model name or alias
            max_tokens: Maximum output tokens
            temperature: Sampling temperature (ignored if thinking enabled)
            thinking: Extended thinking configuration
            **kwargs: Additional API parameters

        Returns:
            LLMResponse with content, usage, and optional thinking output
        """
        resolved_model = self.resolve_model(model)
        model_config = self.get_model_config(model)

        logger.debug(
            "Calling Anthropic API",
            model=resolved_model,
            prompt_length=len(prompt),
            system_length=len(system),
            thinking_enabled=thinking.enabled if thinking else False,
        )

        messages = [{"role": "user", "content": prompt}]

        # Build API parameters
        api_params: dict[str, Any] = {
            "model": resolved_model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        # Add system prompt if provided
        if system:
            api_params["system"] = system

        # Handle thinking mode
        if thinking and thinking.enabled:
            # Thinking mode requires temperature=1 and specific params
            api_params["temperature"] = 1.0
            api_params["thinking"] = thinking.to_api_param()

            # Ensure max_tokens can accommodate thinking + response
            if model_config and model_config.supports_thinking:
                # Thinking budget is separate from max_tokens
                pass
        else:
            api_params["temperature"] = temperature

        # Add any additional kwargs
        api_params.update(kwargs)

        response = await self._client.messages.create(**api_params)

        # Extract content and thinking from response
        content = ""
        thinking_content = ""

        for block in response.content:
            if block.type == "thinking":
                thinking_content = block.thinking
            elif block.type == "text":
                content = block.text

        # Build token usage
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        # Check for cache tokens (if using prompt caching)
        if hasattr(response.usage, "cache_read_input_tokens"):
            usage.cache_read_tokens = response.usage.cache_read_input_tokens or 0
        if hasattr(response.usage, "cache_creation_input_tokens"):
            usage.cache_write_tokens = response.usage.cache_creation_input_tokens or 0

        logger.debug(
            "Anthropic response received",
            model=resolved_model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            thinking_tokens=usage.thinking_tokens,
            stop_reason=response.stop_reason,
            has_thinking=bool(thinking_content),
        )

        return LLMResponse(
            content=content,
            model=resolved_model,
            usage=usage,
            stop_reason=response.stop_reason,
            provider=self.provider_name,
            thinking_content=thinking_content,
        )

    @llm_retry
    @llm_circuit_breaker
    @wrap_anthropic_errors
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
        Stream response from Anthropic API.

        Resilience: Retry with exponential backoff + circuit breaker.
        Note: No timeout on stream as responses can be legitimately long.

        Note: Extended thinking content is not streamed, only the final response.
        For thinking content, use complete() instead.
        """
        resolved_model = self.resolve_model(model)

        logger.debug(
            "Starting Anthropic stream",
            model=resolved_model,
            prompt_length=len(prompt),
            thinking_enabled=thinking.enabled if thinking else False,
        )

        messages = [{"role": "user", "content": prompt}]

        # Build API parameters
        api_params: dict[str, Any] = {
            "model": resolved_model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if system:
            api_params["system"] = system

        # Handle thinking mode for streaming
        if thinking and thinking.enabled:
            api_params["temperature"] = 1.0
            api_params["thinking"] = thinking.to_api_param()
        else:
            api_params["temperature"] = temperature

        api_params.update(kwargs)

        async with self._client.messages.stream(**api_params) as stream:
            async for text in stream.text_stream:
                yield text
