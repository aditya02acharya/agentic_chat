"""High-level document service for API and graph integration."""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any

from agentic_chatbot.documents.config import DocumentConfig
from agentic_chatbot.documents.models import (
    DocumentMetadata,
    DocumentStatus,
    DocumentSummary,
    LoadedDocument,
)
from agentic_chatbot.documents.processor import DocumentProcessor
from agentic_chatbot.documents.storage.base import DocumentStorage
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class DocumentServiceError(Exception):
    """Base exception for document service errors."""
    pass


class DocumentLimitExceeded(DocumentServiceError):
    """Raised when document limit is exceeded."""
    pass


class DocumentNotFound(DocumentServiceError):
    """Raised when document is not found."""
    pass


class DocumentService:
    """
    High-level document operations.

    Provides:
    - Document upload and validation
    - Status tracking and waiting
    - Content loading (full or by chunks)
    - Summary retrieval for supervisor
    """

    def __init__(
        self,
        storage: DocumentStorage,
        llm_client: Any,  # LLMClient type
        config: DocumentConfig | None = None,
    ):
        """
        Initialize document service.

        Args:
            storage: Document storage backend
            llm_client: LLM client for summarization
            config: Document configuration
        """
        self._storage = storage
        self._config = config or DocumentConfig()
        self._processor = DocumentProcessor(storage, llm_client, config)

    async def create_document(
        self,
        conversation_id: str,
        filename: str,
        content: str,
        content_type: str,
    ) -> str:
        """
        Create a new document.

        Args:
            conversation_id: Conversation identifier
            filename: Original filename
            content: Document text content
            content_type: MIME type

        Returns:
            Document ID

        Raises:
            DocumentLimitExceeded: If conversation has max documents
            ValueError: If content type not allowed
        """
        # Check document limit
        current_count = await self._storage.count_documents(conversation_id)
        if current_count >= self._config.MAX_DOCUMENTS_PER_CONVERSATION:
            raise DocumentLimitExceeded(
                f"Maximum {self._config.MAX_DOCUMENTS_PER_CONVERSATION} documents per conversation"
            )

        # Validate content type
        if content_type not in self._config.ALLOWED_CONTENT_TYPES:
            raise ValueError(
                f"Content type '{content_type}' not allowed. "
                f"Allowed types: {self._config.ALLOWED_CONTENT_TYPES}"
            )

        # Generate document ID
        document_id = f"doc-{uuid.uuid4().hex[:12]}"

        # Create metadata
        metadata = DocumentMetadata(
            id=document_id,
            conversation_id=conversation_id,
            filename=filename,
            content_type=content_type,
            size_bytes=len(content.encode("utf-8")),
            char_count=len(content),
            chunk_count=0,  # Set during processing
            uploaded_at=datetime.utcnow(),
            status=DocumentStatus.UPLOADING,
        )

        # Save content and metadata
        await self._storage.save_content(conversation_id, document_id, content)
        await self._storage.save_metadata(conversation_id, document_id, metadata)

        logger.info(
            "Document created",
            conversation_id=conversation_id,
            document_id=document_id,
            filename=filename,
            size=len(content),
        )

        return document_id

    async def process_document(
        self,
        conversation_id: str,
        document_id: str,
    ) -> None:
        """
        Process a document (chunking + summarization).

        This should be called as a background task.

        Args:
            conversation_id: Conversation identifier
            document_id: Document identifier
        """
        await self._processor.process_document(conversation_id, document_id)

    async def wait_for_documents(
        self,
        conversation_id: str,
        document_ids: list[str] | None = None,
        timeout: float | None = None,
    ) -> dict[str, DocumentStatus]:
        """
        Wait for documents to finish processing.

        Args:
            conversation_id: Conversation identifier
            document_ids: Specific docs to wait for (None = all)
            timeout: Max wait time in seconds (uses default if None)

        Returns:
            Dict of document_id -> final status
        """
        timeout = timeout or self._config.DEFAULT_WAIT_TIMEOUT
        poll_interval = self._config.WAIT_POLL_INTERVAL

        start = time.monotonic()

        while (time.monotonic() - start) < timeout:
            statuses = await self._storage.get_document_statuses(
                conversation_id, document_ids
            )

            # Check if all done (READY or ERROR)
            all_done = all(
                s in (DocumentStatus.READY, DocumentStatus.ERROR)
                for s in statuses.values()
            )

            if all_done:
                return statuses

            await asyncio.sleep(poll_interval)

        # Timeout - return current statuses
        return await self._storage.get_document_statuses(
            conversation_id, document_ids
        )

    async def get_summaries(
        self,
        conversation_id: str,
    ) -> list[DocumentSummary]:
        """
        Get all document summaries for supervisor context.

        Args:
            conversation_id: Conversation identifier

        Returns:
            List of document summaries
        """
        return await self._storage.load_all_summaries(conversation_id)

    async def load_full_document(
        self,
        conversation_id: str,
        document_id: str,
    ) -> LoadedDocument | None:
        """
        Load full document content.

        Args:
            conversation_id: Conversation identifier
            document_id: Document identifier

        Returns:
            LoadedDocument or None if not found/not ready
        """
        try:
            metadata = await self._storage.load_metadata(conversation_id, document_id)

            if metadata.status != DocumentStatus.READY:
                logger.warning(
                    "Document not ready for loading",
                    document_id=document_id,
                    status=metadata.status.value,
                )
                return None

            content = await self._storage.load_content(conversation_id, document_id)
            chunks = await self._storage.load_chunks(conversation_id, document_id)

            return LoadedDocument(
                document_id=document_id,
                filename=metadata.filename,
                content=content,
                chunks_loaded=list(range(len(chunks))),
                is_complete=True,
                token_count=sum(c.token_estimate for c in chunks),
            )

        except FileNotFoundError:
            logger.warning(
                "Document not found",
                conversation_id=conversation_id,
                document_id=document_id,
            )
            return None

    async def load_document_chunks(
        self,
        conversation_id: str,
        document_id: str,
        chunk_indices: list[int] | None = None,
    ) -> LoadedDocument | None:
        """
        Load specific chunks from a document.

        Args:
            conversation_id: Conversation identifier
            document_id: Document identifier
            chunk_indices: Specific chunks to load (None = all)

        Returns:
            LoadedDocument or None if not found/not ready
        """
        try:
            metadata = await self._storage.load_metadata(conversation_id, document_id)

            if metadata.status != DocumentStatus.READY:
                logger.warning(
                    "Document not ready for loading",
                    document_id=document_id,
                    status=metadata.status.value,
                )
                return None

            chunks = await self._storage.load_chunks(
                conversation_id, document_id, chunk_indices
            )

            if not chunks:
                return None

            # Concatenate chunk content
            content = "\n\n".join(c.content for c in chunks)
            loaded_indices = [c.chunk_index for c in chunks]
            is_complete = len(loaded_indices) == metadata.chunk_count

            return LoadedDocument(
                document_id=document_id,
                filename=metadata.filename,
                content=content,
                chunks_loaded=loaded_indices,
                is_complete=is_complete,
                token_count=sum(c.token_estimate for c in chunks),
            )

        except FileNotFoundError:
            logger.warning(
                "Document not found",
                conversation_id=conversation_id,
                document_id=document_id,
            )
            return None

    async def get_metadata(
        self,
        conversation_id: str,
        document_id: str,
    ) -> DocumentMetadata:
        """
        Get document metadata.

        Args:
            conversation_id: Conversation identifier
            document_id: Document identifier

        Returns:
            Document metadata

        Raises:
            DocumentNotFound: If document doesn't exist
        """
        try:
            return await self._storage.load_metadata(conversation_id, document_id)
        except FileNotFoundError as e:
            raise DocumentNotFound(
                f"Document not found: {document_id}"
            ) from e

    async def delete_document(
        self,
        conversation_id: str,
        document_id: str,
    ) -> None:
        """
        Delete a document.

        Args:
            conversation_id: Conversation identifier
            document_id: Document identifier
        """
        await self._storage.delete_document(conversation_id, document_id)

        logger.info(
            "Document deleted",
            conversation_id=conversation_id,
            document_id=document_id,
        )

    async def count_documents(
        self,
        conversation_id: str,
    ) -> int:
        """
        Count documents in a conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Number of documents
        """
        return await self._storage.count_documents(conversation_id)
