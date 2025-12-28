"""Document summarization using LLM with resilience patterns."""

import json
from typing import Any

from agentic_chatbot.core.resilience import (
    llm_retry,
    llm_circuit_breaker,
    llm_timeout,
)
from agentic_chatbot.documents.config import DocumentConfig
from agentic_chatbot.documents.models import (
    ChunkSummaryResult,
    DocumentChunk,
    DocumentSummaryResult,
)
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


# Prompt for summarizing a single chunk
CHUNK_SUMMARY_PROMPT = """Analyze this text chunk and provide a concise summary.

TEXT CHUNK ({chunk_index}/{total_chunks}):
{content}

Respond with a JSON object containing:
1. "summary": A 1-2 sentence summary of the key information (max 50 words)
2. "key_topics": A list of 3-5 keywords/topics from this chunk

Be concise and focus on the most important information.

JSON Response:"""


# Prompt for aggregating chunk summaries into document summary
DOCUMENT_SUMMARY_PROMPT = """Based on these chunk summaries, create an overall document summary.

DOCUMENT: {filename}
CHUNK SUMMARIES:
{chunk_summaries}

Respond with a JSON object containing:
1. "overall_summary": A 2-3 sentence summary of the entire document (max 100 words)
2. "key_topics": A list of 5-8 main topics from the document (deduplicated)
3. "document_type": One of: "report", "article", "code", "notes", "email", "data", "documentation", "other"
4. "relevance_hints": When would someone need this document? (1 sentence)

JSON Response:"""


class DocumentSummarizer:
    """
    LLM-based document summarization.

    Uses fast model (haiku) for efficiency.
    Applies resilience patterns for reliability.
    """

    def __init__(
        self,
        llm_client: Any,  # LLMClient type
        config: DocumentConfig | None = None,
    ):
        """
        Initialize summarizer.

        Args:
            llm_client: LLM client for API calls
            config: Document configuration
        """
        self._llm = llm_client
        self._config = config or DocumentConfig()

    @llm_retry
    @llm_circuit_breaker
    @llm_timeout
    async def summarize_chunk(
        self,
        content: str,
        chunk_index: int,
        total_chunks: int,
    ) -> ChunkSummaryResult:
        """
        Summarize a single document chunk.

        Args:
            content: Chunk text content
            chunk_index: Index of this chunk (0-based)
            total_chunks: Total number of chunks

        Returns:
            ChunkSummaryResult with summary and topics
        """
        # Truncate very long chunks for summarization
        max_chars = 6000  # ~1500 tokens
        if len(content) > max_chars:
            content = content[:max_chars] + "...[truncated]"

        prompt = CHUNK_SUMMARY_PROMPT.format(
            chunk_index=chunk_index + 1,  # 1-indexed for display
            total_chunks=total_chunks,
            content=content,
        )

        try:
            response = await self._llm.complete(
                prompt=prompt,
                model=self._config.SUMMARIZATION_MODEL,
                max_tokens=self._config.CHUNK_SUMMARY_MAX_TOKENS,
                temperature=0.0,
            )

            result = self._parse_json_response(response.content)

            return ChunkSummaryResult(
                summary=result.get("summary", ""),
                key_topics=result.get("key_topics", []),
            )

        except Exception as e:
            logger.error(
                "Failed to summarize chunk",
                chunk_index=chunk_index,
                error=str(e),
            )
            # Return empty result on failure
            return ChunkSummaryResult(
                summary=f"[Summary failed: {e}]",
                key_topics=[],
            )

    @llm_retry
    @llm_circuit_breaker
    @llm_timeout
    async def aggregate_summaries(
        self,
        filename: str,
        chunks: list[DocumentChunk],
    ) -> DocumentSummaryResult:
        """
        Aggregate chunk summaries into overall document summary.

        Args:
            filename: Original filename
            chunks: List of chunks with summaries

        Returns:
            DocumentSummaryResult with overall summary
        """
        # Format chunk summaries for prompt
        chunk_summaries = "\n".join(
            f"Chunk {c.chunk_index + 1}: {c.summary}\n  Topics: {', '.join(c.key_topics)}"
            for c in chunks
            if c.summary
        )

        prompt = DOCUMENT_SUMMARY_PROMPT.format(
            filename=filename,
            chunk_summaries=chunk_summaries,
        )

        try:
            response = await self._llm.complete(
                prompt=prompt,
                model=self._config.SUMMARIZATION_MODEL,
                max_tokens=self._config.OVERALL_SUMMARY_MAX_TOKENS,
                temperature=0.0,
            )

            result = self._parse_json_response(response.content)

            return DocumentSummaryResult(
                overall_summary=result.get("overall_summary", ""),
                key_topics=result.get("key_topics", []),
                document_type=result.get("document_type", "unknown"),
                relevance_hints=result.get("relevance_hints", ""),
            )

        except Exception as e:
            logger.error(
                "Failed to aggregate summaries",
                filename=filename,
                error=str(e),
            )
            # Create fallback from chunk topics
            all_topics = []
            for c in chunks:
                all_topics.extend(c.key_topics)
            unique_topics = list(dict.fromkeys(all_topics))[:8]

            return DocumentSummaryResult(
                overall_summary=f"Document containing information about: {', '.join(unique_topics[:5])}",
                key_topics=unique_topics,
                document_type="unknown",
                relevance_hints="",
            )

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """
        Parse JSON from LLM response.

        Handles common formatting issues.
        """
        # Clean up response
        text = response.strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON object
            start = text.find("{")
            end = text.rfind("}") + 1

            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

            logger.warning("Failed to parse JSON response", response=text[:200])
            return {}
