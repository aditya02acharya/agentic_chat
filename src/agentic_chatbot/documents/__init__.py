"""Document upload and context management module.

Provides document upload, chunking, summarization, and context loading
for conversation-attached documents.
"""

from agentic_chatbot.documents.chunker import DocumentChunker
from agentic_chatbot.documents.config import ChunkConfig, DocumentConfig
from agentic_chatbot.documents.models import (
    ChunkSummaryResult,
    DocumentChunk,
    DocumentMetadata,
    DocumentStatus,
    DocumentSummary,
    DocumentSummaryResult,
    LoadedDocument,
)
from agentic_chatbot.documents.processor import DocumentProcessor
from agentic_chatbot.documents.service import (
    DocumentLimitExceeded,
    DocumentNotFound,
    DocumentService,
    DocumentServiceError,
)
from agentic_chatbot.documents.storage import DocumentStorage, LocalDocumentStorage
from agentic_chatbot.documents.summarizer import DocumentSummarizer

__all__ = [
    # Config
    "ChunkConfig",
    "DocumentConfig",
    # Models
    "DocumentStatus",
    "DocumentChunk",
    "DocumentMetadata",
    "DocumentSummary",
    "LoadedDocument",
    "ChunkSummaryResult",
    "DocumentSummaryResult",
    # Storage
    "DocumentStorage",
    "LocalDocumentStorage",
    # Processing
    "DocumentChunker",
    "DocumentSummarizer",
    "DocumentProcessor",
    # Service
    "DocumentService",
    "DocumentServiceError",
    "DocumentLimitExceeded",
    "DocumentNotFound",
]
