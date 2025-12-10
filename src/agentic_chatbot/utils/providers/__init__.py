"""LLM provider implementations.

Supports multiple LLM providers with a unified interface:
- Anthropic: Direct API access to Claude models
- Bedrock: AWS Bedrock access to Claude models

Usage:
    from agentic_chatbot.utils.providers import create_provider

    # Create provider based on settings
    provider = create_provider()

    # Or explicitly create a specific provider
    from agentic_chatbot.utils.providers import AnthropicProvider, BedrockProvider

    anthropic = AnthropicProvider(api_key="...")
    bedrock = BedrockProvider(region_name="us-east-1")
"""

from agentic_chatbot.utils.providers.base import BaseLLMProvider, LLMResponse
from agentic_chatbot.utils.providers.anthropic import AnthropicProvider
from agentic_chatbot.utils.providers.bedrock import BedrockProvider


def create_provider(
    provider: str | None = None,
    **kwargs,
) -> BaseLLMProvider:
    """
    Factory function to create LLM provider based on configuration.

    Args:
        provider: Provider name ("anthropic" or "bedrock"). If None, uses settings.
        **kwargs: Provider-specific arguments

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If unknown provider specified
        ImportError: If provider dependencies not installed

    Example:
        # Use settings (default)
        provider = create_provider()

        # Explicit Anthropic
        provider = create_provider("anthropic", api_key="sk-...")

        # Explicit Bedrock
        provider = create_provider("bedrock", region_name="us-west-2")
    """
    from agentic_chatbot.config.settings import get_settings

    settings = get_settings()

    # Determine provider
    provider_name = provider or settings.llm_provider

    if provider_name == "anthropic":
        api_key = kwargs.get("api_key") or settings.anthropic_api_key
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable."
            )
        return AnthropicProvider(api_key=api_key)

    elif provider_name == "bedrock":
        return BedrockProvider(
            region_name=kwargs.get("region_name") or settings.bedrock_region,
            profile_name=kwargs.get("profile_name") or settings.bedrock_profile,
            aws_access_key_id=kwargs.get("aws_access_key_id") or settings.aws_access_key_id,
            aws_secret_access_key=kwargs.get("aws_secret_access_key") or settings.aws_secret_access_key,
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. "
            f"Supported providers: anthropic, bedrock"
        )


__all__ = [
    "BaseLLMProvider",
    "LLMResponse",
    "AnthropicProvider",
    "BedrockProvider",
    "create_provider",
]
