"""Async LLM client wrapper for Anthropic Claude."""

import os
import time
from dataclasses import dataclass
from typing import AsyncIterator, cast

from anthropic import AsyncAnthropic
from anthropic.types import TextBlock, MessageParam

from ..config.settings import get_settings
from .logging import get_logger

logger = get_logger(__name__)

MODEL_ALIASES = {
    "haiku": "claude-3-haiku-20240307",
    "sonnet": "claude-sonnet-4-20250514",
    "opus": "claude-opus-4-20250514",
}


@dataclass
class LLMResponse:
    """Response from LLM call."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    duration_ms: float


class LLMClient:
    """
    Async wrapper for Anthropic Claude API.

    Features:
    - Model aliases ("haiku", "sonnet", "opus")
    - Usage tracking (tokens, duration)
    - Streaming support
    """

    def __init__(self, api_key: str | None = None):
        settings = get_settings()
        self.api_key = api_key or settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.client = AsyncAnthropic(api_key=self.api_key)
        self.default_model = settings.default_model

    def _resolve_model(self, model: str) -> str:
        """Resolve model alias to full model name."""
        return MODEL_ALIASES.get(model, model)

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str = "sonnet",
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Complete a prompt with Claude.

        Args:
            prompt: User message
            system: System prompt
            model: Model name or alias
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with content and usage info
        """
        resolved_model = self._resolve_model(model)
        start_time = time.time()

        messages: list[MessageParam] = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": resolved_model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        response = await self.client.messages.create(**kwargs)

        duration_ms = (time.time() - start_time) * 1000

        content = ""
        if response.content:
            first_block = response.content[0]
            if isinstance(first_block, TextBlock):
                content = first_block.text

        return LLMResponse(
            content=content,
            model=resolved_model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            duration_ms=duration_ms,
        )

    async def stream(
        self,
        prompt: str,
        system: str = "",
        model: str = "sonnet",
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """
        Stream a completion from Claude.

        Args:
            prompt: User message
            system: System prompt
            model: Model name or alias
            max_tokens: Maximum tokens in response

        Yields:
            Text chunks as they arrive
        """
        resolved_model = self._resolve_model(model)
        messages: list[MessageParam] = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": resolved_model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
