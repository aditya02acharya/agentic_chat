"""Document loading tool for context injection."""

import json
from typing import Any

from agentic_chatbot.mcp.models import (
    MessagingCapabilities,
    OutputDataType,
    ToolResult,
    ToolContent,
    ToolResultStatus,
)
from agentic_chatbot.tools.base import LocalTool, LocalToolContext
from agentic_chatbot.tools.registry import LocalToolRegistry
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


@LocalToolRegistry.register
class LoadDocumentTool(LocalTool):
    """
    Load document content into context for answering questions.

    Use this tool when:
    - User asks about a document they uploaded
    - Document summaries indicate relevant content
    - Full or partial document content is needed

    The tool supports:
    - Loading full documents
    - Loading specific chunks for targeted retrieval
    - Loading multiple documents at once

    Returns document content with metadata for context assembly.
    """

    name = "load_document"
    description = (
        "Load uploaded document content into context. Use when document summaries "
        "indicate the document may contain relevant information for the user's question. "
        "Can load full documents or specific chunks for targeted retrieval."
    )

    input_schema = {
        "type": "object",
        "properties": {
            "document_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of document IDs to load (from document summaries)",
            },
            "chunk_indices": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "description": (
                    "Optional: specific chunks to load per document. "
                    "Format: {document_id: [chunk_index, ...]}. "
                    "If not specified, loads full documents."
                ),
            },
            "wait_if_processing": {
                "type": "boolean",
                "description": (
                    "Wait for documents to finish processing if still in progress. "
                    "Default: true"
                ),
                "default": True,
            },
        },
        "required": ["document_ids"],
    }

    messaging = MessagingCapabilities(
        output_types=[OutputDataType.TEXT, OutputDataType.JSON],
        supports_progress=False,
        supports_elicitation=False,
        supports_direct_response=False,
        supports_streaming=False,
    )

    # This tool needs document service access
    needs_document_service: bool = True

    async def execute(self, context: LocalToolContext) -> ToolResult:
        """Load document(s) content for context injection."""
        # Validate document service is available
        if not context.document_service:
            return self.error(
                f"local:{self.name}",
                "Document service not available. Documents feature may not be enabled.",
            )

        if not context.conversation_id:
            return self.error(
                f"local:{self.name}",
                "Conversation ID required to load documents.",
            )

        document_ids = context.params.get("document_ids", [])
        chunk_indices = context.params.get("chunk_indices", {})
        wait_if_processing = context.params.get("wait_if_processing", True)

        if not document_ids:
            return self.error(
                f"local:{self.name}",
                "No document IDs provided.",
            )

        try:
            doc_service = context.document_service
            conversation_id = context.conversation_id
            loaded_documents = []
            errors = []

            # Wait for documents if requested
            if wait_if_processing:
                from agentic_chatbot.documents.models import DocumentStatus

                statuses = await doc_service.wait_for_documents(
                    conversation_id,
                    document_ids,
                    timeout=30.0,  # Max 30s wait
                )

                # Check for documents still processing
                for doc_id, status in statuses.items():
                    if status not in (DocumentStatus.READY, DocumentStatus.ERROR):
                        errors.append({
                            "document_id": doc_id,
                            "error": f"Document still processing (status: {status.value})",
                        })

            # Load each document
            for doc_id in document_ids:
                try:
                    # Check if we should load specific chunks
                    if doc_id in chunk_indices and chunk_indices[doc_id]:
                        loaded = await doc_service.load_document_chunks(
                            conversation_id,
                            doc_id,
                            chunk_indices[doc_id],
                        )
                    else:
                        loaded = await doc_service.load_full_document(
                            conversation_id,
                            doc_id,
                        )

                    if loaded:
                        loaded_documents.append({
                            "document_id": loaded.document_id,
                            "filename": loaded.filename,
                            "content": loaded.content,
                            "chunks_loaded": loaded.chunks_loaded,
                            "is_complete": loaded.is_complete,
                            "token_count": loaded.token_count,
                        })
                    else:
                        errors.append({
                            "document_id": doc_id,
                            "error": "Document not found or not ready",
                        })

                except Exception as e:
                    logger.error(
                        "Failed to load document",
                        document_id=doc_id,
                        error=str(e),
                    )
                    errors.append({
                        "document_id": doc_id,
                        "error": str(e),
                    })

            # Build result
            result = {
                "loaded_count": len(loaded_documents),
                "documents": loaded_documents,
                "total_tokens": sum(d["token_count"] for d in loaded_documents),
            }

            if errors:
                result["errors"] = errors

            # Format for context - include document content prominently
            if loaded_documents:
                # Create a text representation for the LLM context
                text_parts = [
                    f"Loaded {len(loaded_documents)} document(s) "
                    f"({result['total_tokens']} tokens):\n"
                ]

                for doc in loaded_documents:
                    text_parts.append(f"\n--- {doc['filename']} (ID: {doc['document_id']}) ---")
                    if not doc["is_complete"]:
                        text_parts.append(
                            f"[Partial: chunks {doc['chunks_loaded']}]"
                        )
                    text_parts.append(doc["content"])
                    text_parts.append("--- END ---\n")

                if errors:
                    text_parts.append(f"\nNote: {len(errors)} document(s) could not be loaded.")

                return ToolResult(
                    tool_name=f"local:{self.name}",
                    status=ToolResultStatus.SUCCESS,
                    contents=[
                        ToolContent(
                            content_type="text/plain",
                            data="\n".join(text_parts),
                        ),
                        ToolContent(
                            content_type="application/json",
                            data=json.dumps(result),
                        ),
                    ],
                    duration_ms=0.0,
                )
            else:
                return self.error(
                    f"local:{self.name}",
                    f"No documents could be loaded. Errors: {errors}",
                )

        except Exception as e:
            logger.error(
                "Document loading failed",
                error=str(e),
                exc_info=True,
            )
            return self.error(
                f"local:{self.name}",
                f"Failed to load documents: {e}",
            )


@LocalToolRegistry.register
class ListDocumentsTool(LocalTool):
    """
    List document summaries for a conversation.

    Use this tool to see what documents are available and their summaries.
    The summaries help decide which documents to load for answering questions.
    """

    name = "list_documents"
    description = (
        "List all uploaded documents and their summaries for the current conversation. "
        "Use this to understand what documents are available before loading them."
    )

    input_schema = {
        "type": "object",
        "properties": {
            "include_processing": {
                "type": "boolean",
                "description": "Include documents still being processed. Default: true",
                "default": True,
            },
        },
    }

    messaging = MessagingCapabilities(
        output_types=[OutputDataType.TEXT, OutputDataType.JSON],
        supports_progress=False,
        supports_elicitation=False,
        supports_direct_response=False,
        supports_streaming=False,
    )

    async def execute(self, context: LocalToolContext) -> ToolResult:
        """List document summaries."""
        if not context.document_service:
            return self.error(
                f"local:{self.name}",
                "Document service not available.",
            )

        if not context.conversation_id:
            return self.error(
                f"local:{self.name}",
                "Conversation ID required.",
            )

        try:
            doc_service = context.document_service
            include_processing = context.params.get("include_processing", True)

            summaries = await doc_service.get_summaries(context.conversation_id)

            if not include_processing:
                from agentic_chatbot.documents.models import DocumentStatus
                summaries = [
                    s for s in summaries
                    if s.status == DocumentStatus.READY
                ]

            if not summaries:
                return self.success(
                    f"local:{self.name}",
                    "No documents uploaded for this conversation.",
                )

            # Format summaries
            result = {
                "document_count": len(summaries),
                "documents": [
                    {
                        "document_id": s.document_id,
                        "filename": s.filename,
                        "status": s.status.value,
                        "summary": s.overall_summary,
                        "key_topics": s.key_topics,
                        "document_type": s.document_type,
                        "relevance_hints": s.relevance_hints,
                        "chunk_count": s.chunk_count,
                        "total_tokens": s.total_tokens,
                        "processing_progress": s.processing_progress,
                    }
                    for s in summaries
                ],
            }

            # Create text summary for context
            text_parts = [f"Found {len(summaries)} document(s):\n"]

            for s in summaries:
                text_parts.append(f"\n[{s.document_id}] {s.filename}")
                text_parts.append(f"  Status: {s.status.value}")
                if s.status.value == "ready":
                    text_parts.append(f"  Type: {s.document_type}")
                    text_parts.append(f"  Summary: {s.overall_summary}")
                    text_parts.append(f"  Topics: {', '.join(s.key_topics)}")
                    text_parts.append(f"  Chunks: {s.chunk_count} ({s.total_tokens} tokens)")
                    if s.relevance_hints:
                        text_parts.append(f"  Use when: {s.relevance_hints}")
                else:
                    text_parts.append(f"  Progress: {s.processing_progress:.0%}")

            return ToolResult(
                tool_name=f"local:{self.name}",
                status=ToolResultStatus.SUCCESS,
                contents=[
                    ToolContent(
                        content_type="text/plain",
                        data="\n".join(text_parts),
                    ),
                    ToolContent(
                        content_type="application/json",
                        data=json.dumps(result),
                    ),
                ],
                duration_ms=0.0,
            )

        except Exception as e:
            logger.error(
                "List documents failed",
                error=str(e),
                exc_info=True,
            )
            return self.error(
                f"local:{self.name}",
                f"Failed to list documents: {e}",
            )
