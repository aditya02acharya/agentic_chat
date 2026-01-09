"""Model configuration and registry.

Defines available LLM models with their capabilities, including:
- Thinking mode support (extended thinking)
- Token limits
- Use case categories (fast, balanced, powerful)
- Provider-specific model IDs
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelCategory(str, Enum):
    """Model use case categories."""

    FAST = "fast"  # Quick responses, low latency (haiku-class)
    BALANCED = "balanced"  # Good balance of speed and quality (sonnet-class)
    POWERFUL = "powerful"  # Best quality, complex reasoning (opus-class)
    THINKING = "thinking"  # Extended thinking for complex problems


class ModelCapability(str, Enum):
    """Model capabilities."""

    THINKING = "thinking"  # Supports extended thinking
    VISION = "vision"  # Supports image input
    TOOL_USE = "tool_use"  # Supports function calling
    STREAMING = "streaming"  # Supports streaming responses


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    # Identifiers
    id: str  # Internal identifier (e.g., "claude-3-5-sonnet")
    name: str  # Display name
    aliases: list[str] = field(default_factory=list)  # Short aliases (e.g., ["sonnet"])

    # Provider-specific IDs
    anthropic_id: str = ""  # Anthropic API model ID
    bedrock_id: str = ""  # AWS Bedrock model ID

    # Capabilities
    category: ModelCategory = ModelCategory.BALANCED
    capabilities: list[ModelCapability] = field(default_factory=list)

    # Token limits
    max_output_tokens: int = 4096
    context_window: int = 200000

    # Thinking mode settings
    supports_thinking: bool = False
    default_thinking_budget: int = 10000  # Default thinking tokens if enabled
    max_thinking_budget: int = 50000  # Maximum thinking tokens

    # Knowledge cutoff date (format: "YYYY-MM-DD")
    # This is the date up to which the model has training data
    knowledge_cutoff: str = ""

    # Cost (relative units for optimization)
    cost_per_input_token: float = 1.0
    cost_per_output_token: float = 1.0

    def get_provider_id(self, provider: str) -> str:
        """Get model ID for specific provider."""
        if provider == "anthropic":
            return self.anthropic_id or self.id
        elif provider == "bedrock":
            return self.bedrock_id or f"anthropic.{self.anthropic_id}"
        return self.id


@dataclass
class ThinkingConfig:
    """Configuration for extended thinking mode."""

    enabled: bool = False
    budget_tokens: int = 10000

    def to_api_param(self) -> dict[str, Any] | None:
        """Convert to Anthropic API parameter format."""
        if not self.enabled:
            return None
        return {
            "type": "enabled",
            "budget_tokens": self.budget_tokens,
        }


@dataclass
class TokenUsage:
    """
    Two-level token usage tracking.

    Level 1 - Conversation-level metrics (for UI display):
    - user_input_tokens: Tokens from user's message
    - final_output_tokens: Tokens in final response to user
    - intermediate_tokens: All tokens from agent's intermediate operations
    - total_tokens: Sum of all tokens

    Level 2 - Detailed trace (for OTEL/audit):
    - Tracked via OTEL spans (see telemetry/tracing.py)
    - Each LLM call creates a span with detailed metrics
    """

    # Level 1: Conversation-level aggregates (for UI)
    user_input_tokens: int = 0  # User message input tokens
    final_output_tokens: int = 0  # Final response output tokens

    # Intermediate agent operations (cumulative)
    input_tokens: int = 0  # All input tokens (including intermediate)
    output_tokens: int = 0  # All output tokens (including intermediate)
    thinking_tokens: int = 0  # Extended thinking tokens
    cache_read_tokens: int = 0  # Cached tokens read
    cache_write_tokens: int = 0  # Tokens written to cache

    @property
    def intermediate_tokens(self) -> int:
        """
        Tokens used in intermediate operations (agent reasoning, tool calls, etc).

        Calculated as: total - (user_input + final_output)
        """
        total = self.total_tokens
        user_and_final = self.user_input_tokens + self.final_output_tokens
        return max(0, total - user_and_final)

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all operations."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.thinking_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two token usages together."""
        return TokenUsage(
            user_input_tokens=self.user_input_tokens + other.user_input_tokens,
            final_output_tokens=self.final_output_tokens + other.final_output_tokens,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            thinking_tokens=self.thinking_tokens + other.thinking_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary for API response."""
        return {
            # Level 1: Conversation-level metrics
            "user_input_tokens": self.user_input_tokens,
            "final_output_tokens": self.final_output_tokens,
            "intermediate_tokens": self.intermediate_tokens,
            "total_tokens": self.total_tokens,
            # Detailed breakdown
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }


# =============================================================================
# MODEL REGISTRY
# =============================================================================


class ModelRegistry:
    """Registry of available models."""

    # Define all available models
    MODELS: dict[str, ModelConfig] = {
        # Fast models (haiku-class)
        "claude-3-haiku": ModelConfig(
            id="claude-3-haiku",
            name="Claude 3 Haiku",
            aliases=["haiku", "fast"],
            anthropic_id="claude-3-haiku-20240307",
            bedrock_id="anthropic.claude-3-haiku-20240307-v1:0",
            category=ModelCategory.FAST,
            capabilities=[ModelCapability.VISION, ModelCapability.TOOL_USE, ModelCapability.STREAMING],
            max_output_tokens=4096,
            context_window=200000,
            supports_thinking=False,
            knowledge_cutoff="2023-08-01",  # Claude 3 Haiku training cutoff
            cost_per_input_token=0.25,
            cost_per_output_token=1.25,
        ),
        # Balanced models (sonnet-class)
        "claude-3-5-sonnet": ModelConfig(
            id="claude-3-5-sonnet",
            name="Claude 3.5 Sonnet",
            aliases=["sonnet", "balanced"],
            anthropic_id="claude-3-5-sonnet-20241022",
            bedrock_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
            category=ModelCategory.BALANCED,
            capabilities=[
                ModelCapability.THINKING,
                ModelCapability.VISION,
                ModelCapability.TOOL_USE,
                ModelCapability.STREAMING,
            ],
            max_output_tokens=8192,
            context_window=200000,
            supports_thinking=True,
            default_thinking_budget=10000,
            max_thinking_budget=100000,
            knowledge_cutoff="2024-04-01",  # Claude 3.5 Sonnet training cutoff
            cost_per_input_token=3.0,
            cost_per_output_token=15.0,
        ),
        "claude-3-sonnet": ModelConfig(
            id="claude-3-sonnet",
            name="Claude 3 Sonnet",
            aliases=["sonnet-3"],
            anthropic_id="claude-3-sonnet-20240229",
            bedrock_id="anthropic.claude-3-sonnet-20240229-v1:0",
            category=ModelCategory.BALANCED,
            capabilities=[ModelCapability.VISION, ModelCapability.TOOL_USE, ModelCapability.STREAMING],
            max_output_tokens=4096,
            context_window=200000,
            supports_thinking=False,
            knowledge_cutoff="2023-08-01",  # Claude 3 Sonnet training cutoff
            cost_per_input_token=3.0,
            cost_per_output_token=15.0,
        ),
        # Powerful models (opus-class)
        "claude-3-opus": ModelConfig(
            id="claude-3-opus",
            name="Claude 3 Opus",
            aliases=["opus", "powerful"],
            anthropic_id="claude-3-opus-20240229",
            bedrock_id="anthropic.claude-3-opus-20240229-v1:0",
            category=ModelCategory.POWERFUL,
            capabilities=[ModelCapability.VISION, ModelCapability.TOOL_USE, ModelCapability.STREAMING],
            max_output_tokens=4096,
            context_window=200000,
            supports_thinking=False,
            knowledge_cutoff="2023-08-01",  # Claude 3 Opus training cutoff
            cost_per_input_token=15.0,
            cost_per_output_token=75.0,
        ),
        # Thinking-optimized (for complex reasoning)
        "claude-3-5-sonnet-thinking": ModelConfig(
            id="claude-3-5-sonnet-thinking",
            name="Claude 3.5 Sonnet (Thinking)",
            aliases=["thinking", "reasoning"],
            anthropic_id="claude-3-5-sonnet-20241022",
            bedrock_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
            category=ModelCategory.THINKING,
            capabilities=[
                ModelCapability.THINKING,
                ModelCapability.VISION,
                ModelCapability.TOOL_USE,
                ModelCapability.STREAMING,
            ],
            max_output_tokens=16000,
            context_window=200000,
            supports_thinking=True,
            default_thinking_budget=20000,
            max_thinking_budget=100000,
            knowledge_cutoff="2024-04-01",  # Claude 3.5 Sonnet training cutoff
            cost_per_input_token=3.0,
            cost_per_output_token=15.0,
        ),
    }

    # Build alias lookup
    _alias_map: dict[str, str] = {}

    @classmethod
    def _build_alias_map(cls) -> None:
        """Build alias to model ID mapping."""
        if cls._alias_map:
            return
        for model_id, config in cls.MODELS.items():
            cls._alias_map[model_id] = model_id
            for alias in config.aliases:
                cls._alias_map[alias] = model_id

    @classmethod
    def get(cls, model_id_or_alias: str) -> ModelConfig | None:
        """Get model config by ID or alias."""
        cls._build_alias_map()
        resolved_id = cls._alias_map.get(model_id_or_alias)
        if resolved_id:
            return cls.MODELS.get(resolved_id)
        # Try direct lookup (for full model IDs like "claude-3-5-sonnet-20241022")
        for config in cls.MODELS.values():
            if config.anthropic_id == model_id_or_alias or config.bedrock_id == model_id_or_alias:
                return config
        return None

    @classmethod
    def resolve(cls, model_id_or_alias: str) -> str:
        """Resolve alias to model ID."""
        cls._build_alias_map()
        return cls._alias_map.get(model_id_or_alias, model_id_or_alias)

    @classmethod
    def get_by_category(cls, category: ModelCategory) -> list[ModelConfig]:
        """Get all models in a category."""
        return [m for m in cls.MODELS.values() if m.category == category]

    @classmethod
    def get_thinking_models(cls) -> list[ModelConfig]:
        """Get all models that support thinking."""
        return [m for m in cls.MODELS.values() if m.supports_thinking]

    @classmethod
    def list_all(cls) -> list[ModelConfig]:
        """List all available models."""
        return list(cls.MODELS.values())

    @classmethod
    def list_aliases(cls) -> dict[str, str]:
        """List all aliases and their resolved model IDs."""
        cls._build_alias_map()
        return dict(cls._alias_map)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_model(model_id_or_alias: str) -> ModelConfig:
    """
    Get model config by ID or alias.

    Args:
        model_id_or_alias: Model ID or alias (e.g., "sonnet", "haiku", "thinking")

    Returns:
        ModelConfig for the model

    Raises:
        ValueError: If model not found
    """
    config = ModelRegistry.get(model_id_or_alias)
    if not config:
        available = ", ".join(ModelRegistry.list_aliases().keys())
        raise ValueError(f"Unknown model: {model_id_or_alias}. Available: {available}")
    return config


def get_thinking_config(
    model_id_or_alias: str,
    enable_thinking: bool = False,
    thinking_budget: int | None = None,
) -> ThinkingConfig:
    """
    Get thinking configuration for a model.

    Args:
        model_id_or_alias: Model ID or alias
        enable_thinking: Whether to enable thinking mode
        thinking_budget: Custom thinking budget (uses model default if not specified)

    Returns:
        ThinkingConfig with appropriate settings
    """
    config = get_model(model_id_or_alias)

    if not enable_thinking or not config.supports_thinking:
        return ThinkingConfig(enabled=False)

    budget = thinking_budget or config.default_thinking_budget
    budget = min(budget, config.max_thinking_budget)

    return ThinkingConfig(enabled=True, budget_tokens=budget)


def get_knowledge_cutoff(model_id_or_alias: str) -> str:
    """
    Get the knowledge cutoff date for a model.

    Args:
        model_id_or_alias: Model ID or alias (e.g., "sonnet", "haiku", "thinking")

    Returns:
        Knowledge cutoff date string (e.g., "2024-04-01") or empty string if unknown
    """
    config = ModelRegistry.get(model_id_or_alias)
    if config:
        return config.knowledge_cutoff
    return ""


def get_earliest_knowledge_cutoff(*model_ids: str) -> str:
    """
    Get the earliest knowledge cutoff date among multiple models.

    This is useful for determining the effective knowledge cutoff when
    multiple models are involved in a pipeline (e.g., supervisor + writer).

    Args:
        *model_ids: One or more model IDs or aliases

    Returns:
        The earliest knowledge cutoff date, or empty string if none found
    """
    cutoffs = []
    for model_id in model_ids:
        cutoff = get_knowledge_cutoff(model_id)
        if cutoff:
            cutoffs.append(cutoff)

    if not cutoffs:
        return ""

    # Return the earliest date (lexicographic comparison works for YYYY-MM-DD format)
    return min(cutoffs)
