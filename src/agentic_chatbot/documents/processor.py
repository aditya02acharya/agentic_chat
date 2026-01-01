"""Async document processing pipeline."""

import asyncio
from datetime import datetime
from typing import Any

from agentic_chatbot.documents.chunker import DocumentChunker, estimate_tokens
from agentic_chatbot.documents.config import DocumentConfig
from agentic_chatbot.documents.models import (
    DocumentMetadata,
    DocumentStatus,
    DocumentSummary,
)
from agentic_chatbot.documents.storage.base import DocumentStorage
from agentic_chatbot.documents.summarizer import DocumentSummarizer
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class DocumentProcessor:
    """
    Async document processing with controlled concurrency.

    Processing pipeline:
    1. Load original content
    2. Split into overlapping chunks
    3. Summarize chunks in parallel (with bulkhead)
    4. Aggregate into document summary
    5. Update status to READY
    """

    def __init__(
        self,
        storage: DocumentStorage,
        llm_client: Any,  # LLMClient type
        config: DocumentConfig | None = None,
    ):
        """
        Initialize processor.

        Args:
            storage: Document storage backend
            llm_client: LLM client for summarization
            config: Document configuration
        """
        self._storage = storage
        self._config = config or DocumentConfig()
        self._chunker = DocumentChunker(self._config.chunking)
        self._summarizer = DocumentSummarizer(llm_client, self._config)

        # Semaphore to limit concurrent document processing
        self._processing_semaphore = asyncio.Semaphore(3)

    async def process_document(
        self,
        conversation_id: str,
        document_id: str,
    ) -> None:
        """
        Process a document in background.

        This is the main entry point for document processing.
        Should be called as a background task after upload.

        Args:
            conversation_id: Conversation identifier
            document_id: Document identifier
        """
        async with self._processing_semaphore:
            try:
                await self._process_document_impl(conversation_id, document_id)
            except Exception as e:
                logger.error(
                    "Document processing failed",
                    conversation_id=conversation_id,
                    document_id=document_id,
                    error=str(e),
                )
                await self._storage.update_status(
                    conversation_id,
                    document_id,
                    DocumentStatus.ERROR,
                    error=str(e),
                )

    async def _process_document_impl(
        self,
        conversation_id: str,
        document_id: str,
    ) -> None:
        """Internal processing implementation."""
        logger.info(
            "Starting document processing",
            conversation_id=conversation_id,
            document_id=document_id,
        )

        # Update status to chunking
        await self._storage.update_status(
            conversation_id,
            document_id,
            DocumentStatus.CHUNKING,
            progress=0.1,
        )

        # Load original content
        content = await self._storage.load_content(conversation_id, document_id)
        metadata = await self._storage.load_metadata(conversation_id, document_id)

        # Split into chunks
        chunks = self._chunker.chunk_document(content)

        logger.debug(
            "Document chunked",
            conversation_id=conversation_id,
            document_id=document_id,
            chunk_count=len(chunks),
        )

        # Update metadata with chunk count
        metadata.chunk_count = len(chunks)
        await self._storage.save_metadata(conversation_id, document_id, metadata)

        # Save initial chunks (without summaries)
        await self._storage.save_chunks(conversation_id, document_id, chunks)

        # Update status to summarizing
        await self._storage.update_status(
            conversation_id,
            document_id,
            DocumentStatus.SUMMARIZING,
            progress=0.2,
        )

        # Summarize chunks in parallel with controlled concurrency
        await self._summarize_chunks_parallel(
            conversation_id,
            document_id,
            chunks,
        )

        # Reload chunks (now with summaries)
        chunks = await self._storage.load_chunks(conversation_id, document_id)

        # Aggregate into document summary
        await self._storage.update_status(
            conversation_id,
            document_id,
            DocumentStatus.SUMMARIZING,
            progress=0.9,
        )

        summary_result = await self._summarizer.aggregate_summaries(
            metadata.filename,
            chunks,
        )

        # Create and save document summary
        total_tokens = sum(c.token_estimate for c in chunks)
        chunk_previews = [
            c.summary.split(".")[0] + "." if c.summary else ""
            for c in chunks
        ]

        summary = DocumentSummary(
            document_id=document_id,
            filename=metadata.filename,
            status=DocumentStatus.READY,
            overall_summary=summary_result.overall_summary,
            key_topics=summary_result.key_topics,
            document_type=summary_result.document_type,
            relevance_hints=summary_result.relevance_hints,
            chunk_count=len(chunks),
            chunk_previews=chunk_previews,
            total_tokens=total_tokens,
            processing_progress=1.0,
        )

        await self._storage.save_summary(conversation_id, document_id, summary)

        # Update final status
        await self._storage.update_status(
            conversation_id,
            document_id,
            DocumentStatus.READY,
            progress=1.0,
        )

        logger.info(
            "Document processing complete",
            conversation_id=conversation_id,
            document_id=document_id,
            chunk_count=len(chunks),
            total_tokens=total_tokens,
        )

    async def _summarize_chunks_parallel(
        self,
        conversation_id: str,
        document_id: str,
        chunks: list,
    ) -> None:
        """
        Summarize chunks with controlled concurrency.

        Uses a semaphore to limit concurrent LLM calls.
        """
        # Semaphore for concurrent summarization
        summarize_semaphore = asyncio.Semaphore(
            self._config.MAX_CONCURRENT_SUMMARIZATIONS
        )

        total_chunks = len(chunks)

        async def summarize_one(chunk, index: int):
            async with summarize_semaphore:
                result = await self._summarizer.summarize_chunk(
                    chunk.content,
                    chunk_index=chunk.chunk_index,
                    total_chunks=total_chunks,
                )

                chunk.summary = result.summary
                chunk.key_topics = result.key_topics

                # Update progress
                progress = 0.2 + (0.7 * (index + 1) / total_chunks)
                await self._storage.update_status(
                    conversation_id,
                    document_id,
                    DocumentStatus.SUMMARIZING,
                    progress=progress,
                )

                return chunk

        # Process all chunks concurrently (with semaphore limit)
        tasks = [
            summarize_one(chunk, i)
            for i, chunk in enumerate(chunks)
        ]

        summarized_chunks = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle failures and update chunks
        for i, result in enumerate(summarized_chunks):
            if isinstance(result, Exception):
                logger.error(
                    "Failed to summarize chunk",
                    chunk_index=i,
                    error=str(result),
                )
                chunks[i].summary = f"[Summary failed: {result}]"
                chunks[i].key_topics = []

        # Save updated chunks with summaries
        await self._storage.save_chunks(conversation_id, document_id, chunks)
