"""Episodic Memory: Cross-conversation persistent memory.

This component provides long-term memory across conversations:
- Stores summaries of past interactions
- Retrieves relevant memories for context
- Deduplicates similar memories by merging
- Prunes old/low-importance memories

Key features:
- Topic-based retrieval
- Importance scoring
- Memory consolidation (merging similar memories)
- Automatic pruning with configurable limits
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agentic_chatbot.cognition.config import CognitionSettings, get_cognition_settings
from agentic_chatbot.cognition.models import EpisodicMemory
from agentic_chatbot.utils.logging import get_logger

if TYPE_CHECKING:
    from agentic_chatbot.cognition.storage import CognitionStorage


logger = get_logger(__name__)


# Common stop words to filter from topics
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "just",
    "and", "but", "if", "or", "because", "until", "while", "this",
    "that", "these", "those", "what", "which", "who", "whom",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their",
}


class EpisodicMemoryManager:
    """
    Manager for cross-conversation episodic memory.

    Handles memory creation, retrieval, merging, and pruning
    to maintain a useful set of memories for each user.
    """

    def __init__(
        self,
        storage: CognitionStorage,
        settings: CognitionSettings | None = None,
    ):
        """
        Initialize episodic memory manager.

        Args:
            storage: CognitionStorage for persistence
            settings: Optional settings (uses defaults if not provided)
        """
        self.storage = storage
        self.settings = settings or get_cognition_settings()

    async def get_relevant_memories(
        self,
        user_id: str,
        query: str,
        max_memories: int | None = None,
    ) -> list[EpisodicMemory]:
        """
        Get memories relevant to the current query.

        Uses topic matching to find relevant past interactions.

        Args:
            user_id: User identifier
            query: Current query to match against
            max_memories: Maximum memories to return (uses config default)

        Returns:
            List of relevant memories, ordered by relevance
        """
        max_memories = max_memories or self.settings.max_memories_in_context

        # Extract topics from query
        query_topics = self._extract_topics(query)

        if query_topics:
            # Get topic-matched memories
            memories = await self.storage.get_memories_by_topics(
                user_id=user_id,
                topics=query_topics,
                limit=max_memories,
            )

            # Update access count for retrieved memories
            for memory in memories:
                await self.storage.increment_memory_access(memory.memory_id)

            return memories

        # Fallback to recent important memories if no topics
        return await self.storage.get_memories_for_user(
            user_id=user_id,
            limit=max_memories,
            min_importance=0.3,
        )

    async def create_or_merge_memory(
        self,
        user_id: str,
        conversation_id: str,
        summary: str,
        outcome: str,
        topics: list[str] | None = None,
        importance: float = 0.5,
    ) -> str:
        """
        Create a new memory or merge with existing similar one.

        Checks for similar recent memories and merges if found,
        otherwise creates a new memory.

        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            summary: Summary of the interaction
            outcome: Outcome (success, partial, failure)
            topics: Topics discussed (auto-extracted if not provided)
            importance: Importance score (0-1)

        Returns:
            Memory ID (new or existing merged memory)
        """
        # Extract topics if not provided
        if topics is None:
            topics = self._extract_topics(summary)

        # Look for similar existing memories
        similar = await self.storage.find_similar_memories(
            user_id=user_id,
            topics=topics,
            days=7,  # Look at last week
            limit=5,
        )

        # Check for deduplication candidates
        for existing in similar:
            similarity = self._calculate_similarity(topics, existing.topics)
            if similarity >= self.settings.similarity_threshold:
                # Merge into existing memory
                await self._merge_memories(existing, summary, topics, importance)
                logger.info(
                    f"Merged memory into {existing.memory_id}",
                    similarity=similarity,
                    merged_count=existing.merged_count + 1,
                )
                return existing.memory_id

        # Create new memory
        memory_id = str(uuid.uuid4())
        memory = EpisodicMemory(
            memory_id=memory_id,
            user_id=user_id,
            conversation_id=conversation_id,
            summary=summary,
            outcome=outcome,
            topics=topics,
            importance=importance,
        )

        await self.storage.create_memory(memory)
        logger.info(f"Created new memory {memory_id}", topics=topics)

        return memory_id

    async def prune_memories(self, user_id: str) -> int:
        """
        Prune old/low-importance memories for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of memories deleted
        """
        deleted = await self.storage.prune_old_memories(
            user_id=user_id,
            max_memories=self.settings.max_memories_per_user,
            ttl_days=self.settings.memory_ttl_days,
            low_importance_threshold=self.settings.low_importance_threshold,
        )

        if deleted > 0:
            logger.info(f"Pruned {deleted} memories for user {user_id}")

        return deleted

    def _extract_topics(self, text: str) -> list[str]:
        """
        Extract topic keywords from text.

        Simple keyword extraction focusing on nouns and technical terms.
        """
        # Normalize text
        text = text.lower()

        # Extract words
        words = re.findall(r'\b[a-z]{3,}\b', text)

        # Filter stop words and count frequency
        word_counts: dict[str, int] = {}
        for word in words:
            if word not in STOP_WORDS:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Get top words by frequency
        sorted_words = sorted(
            word_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Return top 10 topics
        return [word for word, _ in sorted_words[:10]]

    def _calculate_similarity(
        self,
        topics1: list[str],
        topics2: list[str],
    ) -> float:
        """
        Calculate Jaccard similarity between topic sets.

        Returns:
            Similarity score between 0 and 1
        """
        if not topics1 or not topics2:
            return 0.0

        set1 = set(t.lower() for t in topics1)
        set2 = set(t.lower() for t in topics2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    async def _merge_memories(
        self,
        existing: EpisodicMemory,
        new_summary: str,
        new_topics: list[str],
        new_importance: float,
    ) -> None:
        """
        Merge new information into an existing memory.

        Updates:
        - Summary (concatenates if different enough)
        - Topics (union of both sets)
        - Importance (increases slightly as reinforcement)
        - Merged count (for tracking)
        """
        # Merge summaries if significantly different
        if new_summary not in existing.summary:
            merged_summary = f"{existing.summary}\n\nAlso: {new_summary[:200]}"
            # Truncate if too long
            if len(merged_summary) > 1000:
                merged_summary = merged_summary[:997] + "..."
        else:
            merged_summary = existing.summary

        # Merge topics (deduplicated)
        merged_topics = list(set(existing.topics + new_topics))[:15]

        # Increase importance (reinforcement)
        merged_importance = min(1.0, existing.importance + 0.1)

        # Update in storage
        await self.storage.update_memory(
            memory_id=existing.memory_id,
            summary=merged_summary,
            topics=merged_topics,
            importance=merged_importance,
            merged_count=existing.merged_count + 1,
        )

    async def summarize_interaction(
        self,
        messages: list[dict[str, Any]],
        outcome: str,
    ) -> str:
        """
        Generate a summary of an interaction for memory storage.

        Simple extraction-based summarization that captures:
        - User's main question/intent
        - Key topics discussed
        - Outcome
        """
        # Extract user messages
        user_messages = [
            m["content"] for m in messages
            if m.get("role") == "user" and m.get("content")
        ]

        if not user_messages:
            return f"Interaction with {outcome} outcome"

        # Take first user message as primary intent
        first_query = user_messages[0][:200]

        # Summarize based on outcome
        if outcome == "success":
            return f"User asked: {first_query}. Successfully answered."
        elif outcome == "partial":
            return f"User asked: {first_query}. Partially addressed."
        else:
            return f"User asked: {first_query}. Could not fully resolve."

    def calculate_importance(
        self,
        messages: list[dict[str, Any]],
        outcome: str,
    ) -> float:
        """
        Calculate importance score for a memory.

        Factors:
        - Outcome (success = higher)
        - Interaction length (longer = more important)
        - Complexity indicators
        """
        base_importance = 0.5

        # Outcome factor
        outcome_scores = {"success": 0.2, "partial": 0.1, "failure": -0.1}
        base_importance += outcome_scores.get(outcome, 0)

        # Length factor (more back-and-forth = more important)
        num_exchanges = len([m for m in messages if m.get("role") == "user"])
        if num_exchanges >= 3:
            base_importance += 0.1
        if num_exchanges >= 5:
            base_importance += 0.1

        # Complexity indicators
        user_text = " ".join(
            m["content"] for m in messages
            if m.get("role") == "user" and m.get("content")
        ).lower()

        complexity_terms = ["workflow", "complex", "multiple", "steps", "plan"]
        if any(term in user_text for term in complexity_terms):
            base_importance += 0.1

        return min(1.0, max(0.0, base_importance))
