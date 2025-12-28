"""Document feature configuration."""

from dataclasses import dataclass


@dataclass
class ChunkConfig:
    """Chunking configuration for accuracy and performance."""

    CHUNK_SIZE: int = 4000           # ~1000 tokens per chunk
    CHUNK_OVERLAP: int = 500         # Overlap to preserve context at boundaries
    MIN_CHUNK_SIZE: int = 500        # Don't create tiny final chunks
    MAX_CHUNKS_PER_DOC: int = 50     # Limit for very large documents

    # Semantic boundaries (prefer splitting at these, in order of preference)
    SPLIT_PATTERNS: tuple[str, ...] = (
        "\n\n\n",      # Triple newline (major section)
        "\n\n",        # Double newline (paragraph)
        "\n",          # Single newline
        ". ",          # Sentence end
        " ",           # Word boundary (last resort)
    )


@dataclass
class DocumentConfig:
    """Document feature configuration."""

    # Limits
    MAX_DOCUMENTS_PER_CONVERSATION: int = 5
    MAX_DOCUMENT_SIZE_BYTES: int = 1_000_000  # 1MB
    ALLOWED_CONTENT_TYPES: tuple[str, ...] = ("text/plain", "text/markdown")

    # Processing
    SUMMARIZATION_MODEL: str = "haiku"
    CHUNK_SUMMARY_MAX_TOKENS: int = 150
    OVERALL_SUMMARY_MAX_TOKENS: int = 300
    MAX_CONCURRENT_SUMMARIZATIONS: int = 5
    PROCESSING_TIMEOUT: float = 120.0  # Max time to process one document

    # Storage
    STORAGE_BACKEND: str = "local"  # "local" or "s3"
    LOCAL_STORAGE_PATH: str = "./storage/documents"
    S3_BUCKET: str = ""
    S3_PREFIX: str = "documents/"

    # Context
    DOCUMENT_CONTEXT_PRIORITY: int = 900

    # Wait settings
    DEFAULT_WAIT_TIMEOUT: float = 30.0
    WAIT_POLL_INTERVAL: float = 0.5

    # Chunking
    chunking: ChunkConfig = None  # type: ignore

    def __post_init__(self) -> None:
        """Initialize nested configs."""
        if self.chunking is None:
            self.chunking = ChunkConfig()


# Default configuration instance
default_config = DocumentConfig()
