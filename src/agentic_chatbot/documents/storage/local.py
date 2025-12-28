"""Local filesystem document storage implementation."""

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path

import aiofiles
import aiofiles.os

from agentic_chatbot.documents.models import (
    DocumentChunk,
    DocumentMetadata,
    DocumentStatus,
    DocumentSummary,
)
from agentic_chatbot.documents.storage.base import DocumentStorage
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class LocalDocumentStorage(DocumentStorage):
    """
    Local filesystem document storage.

    Directory structure:
        {base_path}/
        └── {conversation_id}/
            └── {document_id}/
                ├── original.txt
                ├── metadata.json
                ├── summary.json
                └── chunks/
                    ├── 000.json
                    ├── 001.json
                    └── ...
    """

    def __init__(self, base_path: str = "./storage/documents"):
        """
        Initialize local storage.

        Args:
            base_path: Base directory for document storage
        """
        self._base_path = Path(base_path)
        self._lock = asyncio.Lock()

    def _get_doc_path(self, conversation_id: str, document_id: str) -> Path:
        """Get path to document directory."""
        return self._base_path / conversation_id / document_id

    def _get_chunks_path(self, conversation_id: str, document_id: str) -> Path:
        """Get path to chunks directory."""
        return self._get_doc_path(conversation_id, document_id) / "chunks"

    async def _ensure_dir(self, path: Path) -> None:
        """Ensure directory exists."""
        path.mkdir(parents=True, exist_ok=True)

    async def save_content(
        self,
        conversation_id: str,
        document_id: str,
        content: str,
    ) -> None:
        """Save original document content."""
        doc_path = self._get_doc_path(conversation_id, document_id)
        await self._ensure_dir(doc_path)

        content_file = doc_path / "original.txt"
        async with aiofiles.open(content_file, "w", encoding="utf-8") as f:
            await f.write(content)

        logger.debug(
            "Saved document content",
            conversation_id=conversation_id,
            document_id=document_id,
            size=len(content),
        )

    async def load_content(
        self,
        conversation_id: str,
        document_id: str,
    ) -> str:
        """Load original document content."""
        content_file = self._get_doc_path(conversation_id, document_id) / "original.txt"

        if not content_file.exists():
            raise FileNotFoundError(
                f"Document content not found: {conversation_id}/{document_id}"
            )

        async with aiofiles.open(content_file, "r", encoding="utf-8") as f:
            return await f.read()

    async def save_metadata(
        self,
        conversation_id: str,
        document_id: str,
        metadata: DocumentMetadata,
    ) -> None:
        """Save document metadata."""
        doc_path = self._get_doc_path(conversation_id, document_id)
        await self._ensure_dir(doc_path)

        metadata_file = doc_path / "metadata.json"
        async with aiofiles.open(metadata_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(metadata.to_dict(), indent=2))

    async def load_metadata(
        self,
        conversation_id: str,
        document_id: str,
    ) -> DocumentMetadata:
        """Load document metadata."""
        metadata_file = (
            self._get_doc_path(conversation_id, document_id) / "metadata.json"
        )

        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Document metadata not found: {conversation_id}/{document_id}"
            )

        async with aiofiles.open(metadata_file, "r", encoding="utf-8") as f:
            data = json.loads(await f.read())
            return DocumentMetadata.from_dict(data)

    async def update_status(
        self,
        conversation_id: str,
        document_id: str,
        status: DocumentStatus,
        progress: float = 0.0,
        error: str | None = None,
    ) -> None:
        """Update document processing status."""
        async with self._lock:
            try:
                metadata = await self.load_metadata(conversation_id, document_id)
            except FileNotFoundError:
                logger.warning(
                    "Cannot update status - metadata not found",
                    conversation_id=conversation_id,
                    document_id=document_id,
                )
                return

            metadata.status = status
            metadata.processing_progress = progress
            if error:
                metadata.error_message = error
            if status == DocumentStatus.READY:
                metadata.processed_at = datetime.utcnow()

            await self.save_metadata(conversation_id, document_id, metadata)

            logger.debug(
                "Updated document status",
                conversation_id=conversation_id,
                document_id=document_id,
                status=status.value,
                progress=progress,
            )

    async def save_chunks(
        self,
        conversation_id: str,
        document_id: str,
        chunks: list[DocumentChunk],
    ) -> None:
        """Save document chunks."""
        chunks_path = self._get_chunks_path(conversation_id, document_id)
        await self._ensure_dir(chunks_path)

        # Save each chunk as a separate file
        for chunk in chunks:
            chunk_file = chunks_path / f"{chunk.chunk_index:03d}.json"
            async with aiofiles.open(chunk_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(chunk.to_dict(), indent=2))

        logger.debug(
            "Saved document chunks",
            conversation_id=conversation_id,
            document_id=document_id,
            chunk_count=len(chunks),
        )

    async def load_chunks(
        self,
        conversation_id: str,
        document_id: str,
        chunk_indices: list[int] | None = None,
    ) -> list[DocumentChunk]:
        """Load document chunks."""
        chunks_path = self._get_chunks_path(conversation_id, document_id)

        if not chunks_path.exists():
            return []

        chunks = []

        # Find all chunk files
        chunk_files = sorted(chunks_path.glob("*.json"))

        for chunk_file in chunk_files:
            # Parse chunk index from filename
            try:
                chunk_idx = int(chunk_file.stem)
            except ValueError:
                continue

            # Filter by requested indices
            if chunk_indices is not None and chunk_idx not in chunk_indices:
                continue

            async with aiofiles.open(chunk_file, "r", encoding="utf-8") as f:
                data = json.loads(await f.read())
                chunks.append(DocumentChunk.from_dict(data))

        # Sort by chunk index
        chunks.sort(key=lambda c: c.chunk_index)

        return chunks

    async def save_summary(
        self,
        conversation_id: str,
        document_id: str,
        summary: DocumentSummary,
    ) -> None:
        """Save document summary."""
        doc_path = self._get_doc_path(conversation_id, document_id)
        await self._ensure_dir(doc_path)

        summary_file = doc_path / "summary.json"
        async with aiofiles.open(summary_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(summary.to_dict(), indent=2))

        logger.debug(
            "Saved document summary",
            conversation_id=conversation_id,
            document_id=document_id,
        )

    async def load_summary(
        self,
        conversation_id: str,
        document_id: str,
    ) -> DocumentSummary | None:
        """Load document summary."""
        summary_file = (
            self._get_doc_path(conversation_id, document_id) / "summary.json"
        )

        if not summary_file.exists():
            return None

        async with aiofiles.open(summary_file, "r", encoding="utf-8") as f:
            data = json.loads(await f.read())
            return DocumentSummary.from_dict(data)

    async def load_all_summaries(
        self,
        conversation_id: str,
    ) -> list[DocumentSummary]:
        """Load all document summaries for a conversation."""
        conv_path = self._base_path / conversation_id

        if not conv_path.exists():
            return []

        summaries = []

        # Iterate through document directories
        for doc_dir in conv_path.iterdir():
            if not doc_dir.is_dir():
                continue

            document_id = doc_dir.name

            # Try to load summary
            summary = await self.load_summary(conversation_id, document_id)

            if summary:
                summaries.append(summary)
            else:
                # Create minimal summary from metadata
                try:
                    metadata = await self.load_metadata(conversation_id, document_id)
                    summaries.append(
                        DocumentSummary(
                            document_id=document_id,
                            filename=metadata.filename,
                            status=metadata.status,
                            processing_progress=metadata.processing_progress,
                            error_message=metadata.error_message,
                        )
                    )
                except FileNotFoundError:
                    pass

        return summaries

    async def get_document_statuses(
        self,
        conversation_id: str,
        document_ids: list[str] | None = None,
    ) -> dict[str, DocumentStatus]:
        """Get processing status of documents."""
        if document_ids is None:
            document_ids = await self.list_document_ids(conversation_id)

        statuses = {}

        for doc_id in document_ids:
            try:
                metadata = await self.load_metadata(conversation_id, doc_id)
                statuses[doc_id] = metadata.status
            except FileNotFoundError:
                statuses[doc_id] = DocumentStatus.ERROR

        return statuses

    async def count_documents(
        self,
        conversation_id: str,
    ) -> int:
        """Count documents in a conversation."""
        conv_path = self._base_path / conversation_id

        if not conv_path.exists():
            return 0

        count = 0
        for item in conv_path.iterdir():
            if item.is_dir():
                count += 1

        return count

    async def delete_document(
        self,
        conversation_id: str,
        document_id: str,
    ) -> None:
        """Delete a document and all its data."""
        doc_path = self._get_doc_path(conversation_id, document_id)

        if doc_path.exists():
            # Use shutil.rmtree for recursive deletion
            await asyncio.get_event_loop().run_in_executor(
                None, shutil.rmtree, doc_path
            )

            logger.info(
                "Deleted document",
                conversation_id=conversation_id,
                document_id=document_id,
            )

    async def list_document_ids(
        self,
        conversation_id: str,
    ) -> list[str]:
        """List all document IDs in a conversation."""
        conv_path = self._base_path / conversation_id

        if not conv_path.exists():
            return []

        return [
            item.name
            for item in conv_path.iterdir()
            if item.is_dir()
        ]
