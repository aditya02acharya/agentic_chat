"""Application settings via environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings via environment variables."""

    # LLM Provider Selection
    # Options: "anthropic" (direct API) or "bedrock" (AWS Bedrock)
    llm_provider: Literal["anthropic", "bedrock"] = "anthropic"

    # Anthropic Direct API
    anthropic_api_key: str = ""
    default_model: str = "claude-3-haiku-20240307"
    smart_model: str = "claude-3-5-sonnet-20241022"

    # AWS Bedrock Configuration
    bedrock_region: str = "us-east-1"
    bedrock_profile: str | None = None  # AWS profile name (optional)
    aws_access_key_id: str | None = None  # Optional, uses default credential chain
    aws_secret_access_key: str | None = None

    # MCP
    mcp_discovery_url: str = "http://localhost:8080/servers"
    mcp_timeout_seconds: int = 30
    mcp_cache_ttl_seconds: int = 300  # 5 minutes
    mcp_max_concurrent_per_server: int = 10  # Semaphore limit per server

    # Supervisor
    max_iterations: int = 5

    # Context
    conversation_window_size: int = 5
    max_context_tokens: int = 4000

    # Logging
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
