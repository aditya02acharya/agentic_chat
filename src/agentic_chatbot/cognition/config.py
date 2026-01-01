"""Configuration for the meta-cognitive layer.

Settings for PostgreSQL connection, memory pruning thresholds,
and background task queue behavior.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class CognitionSettings(BaseSettings):
    """Settings for System 3 meta-cognitive layer."""

    # PostgreSQL connection
    cognition_db_host: str = "localhost"
    cognition_db_port: int = 5432
    cognition_db_name: str = "agentic_chatbot"
    cognition_db_user: str = "postgres"
    cognition_db_password: str = ""
    cognition_db_pool_size: int = 5
    cognition_db_pool_max_overflow: int = 10

    # Memory settings
    max_memories_per_user: int = 100
    memory_ttl_days: int = 90
    low_importance_threshold: float = 0.2
    similarity_threshold: float = 0.7  # For deduplication

    # Task queue settings
    task_poll_interval_seconds: float = 1.0
    task_max_retries: int = 3
    task_worker_shutdown_timeout: float = 30.0

    # Context retrieval
    max_memories_in_context: int = 5
    context_load_timeout_ms: int = 100  # Fast timeout for sync loading

    # Feature flags
    cognition_enabled: bool = True
    theory_of_mind_enabled: bool = True
    episodic_memory_enabled: bool = True
    identity_enabled: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def database_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return (
            f"postgresql://{self.cognition_db_user}:{self.cognition_db_password}"
            f"@{self.cognition_db_host}:{self.cognition_db_port}/{self.cognition_db_name}"
        )

    @property
    def async_database_url(self) -> str:
        """Get async PostgreSQL connection URL (asyncpg)."""
        return (
            f"postgresql+asyncpg://{self.cognition_db_user}:{self.cognition_db_password}"
            f"@{self.cognition_db_host}:{self.cognition_db_port}/{self.cognition_db_name}"
        )


@lru_cache
def get_cognition_settings() -> CognitionSettings:
    """Get cached cognition settings instance."""
    return CognitionSettings()
