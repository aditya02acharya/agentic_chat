"""Document chunking with semantic boundary detection and overlap."""

from agentic_chatbot.documents.config import ChunkConfig
from agentic_chatbot.documents.models import DocumentChunk
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses a simple heuristic: ~4 characters per token for English text.
    """
    return len(text) // 4


class DocumentChunker:
    """
    Split documents into overlapping chunks at semantic boundaries.

    Features:
    - Prefers splitting at paragraph/sentence boundaries
    - Maintains overlap between chunks for context preservation
    - Handles edge cases (small documents, very long lines)
    """

    def __init__(self, config: ChunkConfig | None = None):
        """
        Initialize chunker.

        Args:
            config: Chunking configuration (uses defaults if not provided)
        """
        self._config = config or ChunkConfig()

    def chunk_document(self, content: str) -> list[DocumentChunk]:
        """
        Split document into overlapping chunks.

        Args:
            content: Full document text

        Returns:
            List of DocumentChunk objects
        """
        # Small documents don't need chunking
        if len(content) <= self._config.CHUNK_SIZE:
            return [
                DocumentChunk(
                    chunk_index=0,
                    content=content,
                    start_char=0,
                    end_char=len(content),
                    token_estimate=estimate_tokens(content),
                )
            ]

        chunks = []
        current_pos = 0
        chunk_index = 0

        while current_pos < len(content):
            # Calculate chunk boundaries
            chunk_end = min(current_pos + self._config.CHUNK_SIZE, len(content))

            # If not at the end, find a good split point
            if chunk_end < len(content):
                split_pos = self._find_split_point(content, current_pos, chunk_end)
            else:
                split_pos = chunk_end

            # Extract chunk content
            chunk_content = content[current_pos:split_pos]

            # Skip empty chunks
            if chunk_content.strip():
                chunks.append(
                    DocumentChunk(
                        chunk_index=chunk_index,
                        content=chunk_content,
                        start_char=current_pos,
                        end_char=split_pos,
                        token_estimate=estimate_tokens(chunk_content),
                    )
                )
                chunk_index += 1

            # Check if we've hit the max chunks limit
            if chunk_index >= self._config.MAX_CHUNKS_PER_DOC:
                logger.warning(
                    "Document exceeded max chunks, truncating",
                    max_chunks=self._config.MAX_CHUNKS_PER_DOC,
                    total_chars=len(content),
                    processed_chars=split_pos,
                )
                break

            # Move position forward, accounting for overlap
            # Overlap starts from the end of current chunk
            if split_pos >= len(content):
                break

            # Calculate next position with overlap
            overlap_start = max(
                split_pos - self._config.CHUNK_OVERLAP,
                current_pos + self._config.MIN_CHUNK_SIZE,
            )
            current_pos = overlap_start

        # Handle case where final chunk is too small
        if len(chunks) > 1:
            last_chunk = chunks[-1]
            if len(last_chunk.content) < self._config.MIN_CHUNK_SIZE:
                # Merge with previous chunk
                prev_chunk = chunks[-2]
                merged_content = content[prev_chunk.start_char : last_chunk.end_char]
                chunks[-2] = DocumentChunk(
                    chunk_index=prev_chunk.chunk_index,
                    content=merged_content,
                    start_char=prev_chunk.start_char,
                    end_char=last_chunk.end_char,
                    token_estimate=estimate_tokens(merged_content),
                )
                chunks.pop()

        logger.debug(
            "Chunked document",
            total_chars=len(content),
            chunk_count=len(chunks),
            avg_chunk_size=sum(len(c.content) for c in chunks) // max(len(chunks), 1),
        )

        return chunks

    def _find_split_point(
        self,
        content: str,
        start: int,
        target_end: int,
    ) -> int:
        """
        Find the best split point near target_end.

        Prefers splitting at semantic boundaries (paragraphs, sentences).

        Args:
            content: Full document text
            start: Start of current chunk
            target_end: Ideal end position

        Returns:
            Best split position
        """
        # Search window: look backwards from target_end
        search_start = max(start + self._config.MIN_CHUNK_SIZE, target_end - 500)
        search_text = content[search_start:target_end]

        # Try each split pattern in order of preference
        for pattern in self._config.SPLIT_PATTERNS:
            # Find last occurrence of pattern in search window
            last_pos = search_text.rfind(pattern)

            if last_pos != -1:
                # Return position after the pattern
                split_pos = search_start + last_pos + len(pattern)

                # Make sure we're not creating a tiny chunk
                if split_pos - start >= self._config.MIN_CHUNK_SIZE:
                    return split_pos

        # No good boundary found, split at target
        return target_end

    def get_chunk_boundaries(self, content: str) -> list[tuple[int, int]]:
        """
        Get chunk boundaries without creating full chunks.

        Useful for planning or visualization.

        Args:
            content: Full document text

        Returns:
            List of (start, end) tuples
        """
        chunks = self.chunk_document(content)
        return [(c.start_char, c.end_char) for c in chunks]
