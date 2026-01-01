"""Document data models for context management.

This module defines the core data structures for document upload,
processing, and context loading.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class DocumentStatus(str, Enum):
    """Document processing status."""

    UPLOADING = "uploading"      # File being received
    CHUNKING = "chunking"        # Splitting into chunks
    SUMMARIZING = "summarizing"  # LLM processing chunks
    READY = "ready"              # Fully processed
    ERROR = "error"              # Processing failed


@dataclass
class DocumentChunk:
    """A chunk of a document with its summary."""

    chunk_index: int
    content: str                    # Chunk text
    start_char: int                 # Position in original document
    end_char: int
    token_estimate: int             # For context budgeting

    # Populated after summarization
    summary: str = ""               # LLM-generated summary (~50-100 tokens)
    key_topics: list[str] = field(default_factory=list)  # 3-5 keywords

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "chunk_index": self.chunk_index,
            "content": self.content,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "token_estimate": self.token_estimate,
            "summary": self.summary,
            "key_topics": self.key_topics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentChunk":
        """Deserialize from dictionary."""
        return cls(
            chunk_index=data["chunk_index"],
            content=data["content"],
            start_char=data["start_char"],
            end_char=data["end_char"],
            token_estimate=data["token_estimate"],
            summary=data.get("summary", ""),
            key_topics=data.get("key_topics", []),
        )


@dataclass
class DocumentMetadata:
    """Document metadata with processing status."""

    id: str
    conversation_id: str
    filename: str
    content_type: str               # "text/plain", "text/markdown"
    size_bytes: int
    char_count: int
    chunk_count: int
    uploaded_at: datetime
    status: DocumentStatus
    processed_at: datetime | None = None
    error_message: str | None = None
    processing_progress: float = 0.0  # 0.0-1.0 for UI progress bar

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "filename": self.filename,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "char_count": self.char_count,
            "chunk_count": self.chunk_count,
            "uploaded_at": self.uploaded_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "status": self.status.value,
            "error_message": self.error_message,
            "processing_progress": self.processing_progress,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentMetadata":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            conversation_id=data["conversation_id"],
            filename=data["filename"],
            content_type=data["content_type"],
            size_bytes=data["size_bytes"],
            char_count=data["char_count"],
            chunk_count=data["chunk_count"],
            uploaded_at=datetime.fromisoformat(data["uploaded_at"]),
            processed_at=(
                datetime.fromisoformat(data["processed_at"])
                if data.get("processed_at")
                else None
            ),
            status=DocumentStatus(data["status"]),
            error_message=data.get("error_message"),
            processing_progress=data.get("processing_progress", 0.0),
        )


@dataclass
class DocumentSummary:
    """Document summary for supervisor context."""

    document_id: str
    filename: str
    status: DocumentStatus

    # Available when READY
    overall_summary: str = ""           # Aggregated from chunk summaries
    key_topics: list[str] = field(default_factory=list)  # Combined and deduplicated
    document_type: str = "unknown"      # "report", "code", "notes", etc.
    relevance_hints: str = ""           # When to use this document

    # For targeted loading
    chunk_count: int = 0
    chunk_previews: list[str] = field(default_factory=list)  # First line of each chunk summary
    total_tokens: int = 0               # Total document size in tokens

    # Processing info
    processing_progress: float = 0.0
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "status": self.status.value,
            "overall_summary": self.overall_summary,
            "key_topics": self.key_topics,
            "document_type": self.document_type,
            "relevance_hints": self.relevance_hints,
            "chunk_count": self.chunk_count,
            "chunk_previews": self.chunk_previews,
            "total_tokens": self.total_tokens,
            "processing_progress": self.processing_progress,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentSummary":
        """Deserialize from dictionary."""
        return cls(
            document_id=data["document_id"],
            filename=data["filename"],
            status=DocumentStatus(data["status"]),
            overall_summary=data.get("overall_summary", ""),
            key_topics=data.get("key_topics", []),
            document_type=data.get("document_type", "unknown"),
            relevance_hints=data.get("relevance_hints", ""),
            chunk_count=data.get("chunk_count", 0),
            chunk_previews=data.get("chunk_previews", []),
            total_tokens=data.get("total_tokens", 0),
            processing_progress=data.get("processing_progress", 0.0),
            error_message=data.get("error_message"),
        )


@dataclass
class LoadedDocument:
    """Full document content loaded into context."""

    document_id: str
    filename: str
    content: str                    # Full text OR concatenated chunks
    chunks_loaded: list[int]        # Which chunks are included
    is_complete: bool               # True if all chunks loaded
    token_count: int
    priority: int = 900             # Higher = keep in context longer (documents are high priority)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "content": self.content,
            "chunks_loaded": self.chunks_loaded,
            "is_complete": self.is_complete,
            "token_count": self.token_count,
            "priority": self.priority,
        }


@dataclass
class ChunkSummaryResult:
    """Result from summarizing a single chunk."""

    summary: str
    key_topics: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkSummaryResult":
        """Deserialize from dictionary."""
        return cls(
            summary=data.get("summary", ""),
            key_topics=data.get("key_topics", []),
        )


@dataclass
class DocumentSummaryResult:
    """Result from aggregating chunk summaries into document summary."""

    overall_summary: str
    key_topics: list[str]
    document_type: str
    relevance_hints: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentSummaryResult":
        """Deserialize from dictionary."""
        return cls(
            overall_summary=data.get("overall_summary", ""),
            key_topics=data.get("key_topics", []),
            document_type=data.get("document_type", "unknown"),
            relevance_hints=data.get("relevance_hints", ""),
        )
