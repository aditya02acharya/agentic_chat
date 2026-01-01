"""Abstract base class for document storage."""

from abc import ABC, abstractmethod

from agentic_chatbot.documents.models import (
    DocumentChunk,
    DocumentMetadata,
    DocumentStatus,
    DocumentSummary,
)


class DocumentStorage(ABC):
    """
    Abstract document storage interface.

    Implementations can use local filesystem, S3, or other storage backends.

    Storage structure:
        {base_path}/
        └── {conversation_id}/
            ├── manifest.json                    # Quick index of all documents
            └── {document_id}/
                ├── original.txt                 # Full original content
                ├── metadata.json                # Upload info + status
                ├── summary.json                 # Overall summary (when ready)
                └── chunks/
                    ├── index.json               # Chunk manifest
                    ├── 000.json                 # Individual chunks
                    └── ...
    """

    @abstractmethod
    async def save_content(
        self,
        conversation_id: str,
        document_id: str,
        content: str,
    ) -> None:
        """
        Save original document content.

        Args:
            conversation_id: Conversation identifier
            document_id: Document identifier
            content: Full document text
        """
        ...

    @abstractmethod
    async def load_content(
        self,
        conversation_id: str,
        document_id: str,
    ) -> str:
        """
        Load original document content.

        Args:
            conversation_id: Conversation identifier
            document_id: Document identifier

        Returns:
            Full document text

        Raises:
            FileNotFoundError: If document doesn't exist
        """
        ...

    @abstractmethod
    async def save_metadata(
        self,
        conversation_id: str,
        document_id: str,
        metadata: DocumentMetadata,
    ) -> None:
        """
        Save document metadata.

        Args:
            conversation_id: Conversation identifier
            document_id: Document identifier
            metadata: Document metadata
        """
        ...

    @abstractmethod
    async def load_metadata(
        self,
        conversation_id: str,
        document_id: str,
    ) -> DocumentMetadata:
        """
        Load document metadata.

        Args:
            conversation_id: Conversation identifier
            document_id: Document identifier

        Returns:
            Document metadata

        Raises:
            FileNotFoundError: If metadata doesn't exist
        """
        ...

    @abstractmethod
    async def update_status(
        self,
        conversation_id: str,
        document_id: str,
        status: DocumentStatus,
        progress: float = 0.0,
        error: str | None = None,
    ) -> None:
        """
        Update document processing status.

        Args:
            conversation_id: Conversation identifier
            document_id: Document identifier
            status: New status
            progress: Processing progress (0.0-1.0)
            error: Error message if status is ERROR
        """
        ...

    @abstractmethod
    async def save_chunks(
        self,
        conversation_id: str,
        document_id: str,
        chunks: list[DocumentChunk],
    ) -> None:
        """
        Save document chunks.

        Args:
            conversation_id: Conversation identifier
            document_id: Document identifier
            chunks: List of document chunks
        """
        ...

    @abstractmethod
    async def load_chunks(
        self,
        conversation_id: str,
        document_id: str,
        chunk_indices: list[int] | None = None,
    ) -> list[DocumentChunk]:
        """
        Load document chunks.

        Args:
            conversation_id: Conversation identifier
            document_id: Document identifier
            chunk_indices: Specific chunks to load (None = all)

        Returns:
            List of document chunks
        """
        ...

    @abstractmethod
    async def save_summary(
        self,
        conversation_id: str,
        document_id: str,
        summary: DocumentSummary,
    ) -> None:
        """
        Save document summary.

        Args:
            conversation_id: Conversation identifier
            document_id: Document identifier
            summary: Document summary
        """
        ...

    @abstractmethod
    async def load_summary(
        self,
        conversation_id: str,
        document_id: str,
    ) -> DocumentSummary | None:
        """
        Load document summary.

        Args:
            conversation_id: Conversation identifier
            document_id: Document identifier

        Returns:
            Document summary or None if not yet created
        """
        ...

    @abstractmethod
    async def load_all_summaries(
        self,
        conversation_id: str,
    ) -> list[DocumentSummary]:
        """
        Load all document summaries for a conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            List of document summaries
        """
        ...

    @abstractmethod
    async def get_document_statuses(
        self,
        conversation_id: str,
        document_ids: list[str] | None = None,
    ) -> dict[str, DocumentStatus]:
        """
        Get processing status of documents.

        Args:
            conversation_id: Conversation identifier
            document_ids: Specific documents (None = all)

        Returns:
            Dict of document_id -> status
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    async def delete_document(
        self,
        conversation_id: str,
        document_id: str,
    ) -> None:
        """
        Delete a document and all its data.

        Args:
            conversation_id: Conversation identifier
            document_id: Document identifier
        """
        ...

    @abstractmethod
    async def list_document_ids(
        self,
        conversation_id: str,
    ) -> list[str]:
        """
        List all document IDs in a conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            List of document IDs
        """
        ...
