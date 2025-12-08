"""Application settings via environment variables."""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings via environment variables."""

    anthropic_api_key: str = ""
    default_model: str = "claude-3-haiku-20240307"
    smart_model: str = "claude-sonnet-4-20250514"

    mcp_discovery_url: str = "http://localhost:8080/servers"
    mcp_timeout_seconds: int = 30
    mcp_cache_ttl_seconds: int = 300
    mcp_max_concurrent_per_server: int = 10

    max_iterations: int = 5

    conversation_window_size: int = 5
    max_context_tokens: int = 4000

    log_level: str = "INFO"

    host: str = "0.0.0.0"
    port: int = 5000

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
