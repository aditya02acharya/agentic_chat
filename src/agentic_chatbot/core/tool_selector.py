"""Tool Selector - Efficient Tool Filtering.

Tools consume tokens when their schemas are sent to LLMs.
This module provides an abstraction layer that:
1. Filters to top N candidate tools based on query
2. Uses metadata for efficient matching
3. Provides a lightweight tool summary for initial selection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from agentic_chatbot.core.query_understanding import QueryUnderstanding, QueryIntent


class ToolMetadata(BaseModel):
    """
    Lightweight metadata for tool selection.

    This is much smaller than full tool schemas and can be
    used for efficient filtering before loading full schemas.
    """

    tool_id: str = Field(description="Unique tool identifier")
    name: str = Field(description="Tool name")
    description: str = Field(description="Brief description (< 100 chars)")

    # Categorization
    category: str = Field(default="general", description="Tool category")
    subcategory: str = Field(default="", description="More specific category")
    tags: list[str] = Field(default_factory=list, description="Searchable tags")

    # Capabilities
    supported_intents: list[str] = Field(default_factory=list, description="What intents this tool serves")
    input_types: list[str] = Field(default_factory=list, description="What input types it accepts")
    output_types: list[str] = Field(default_factory=list, description="What output types it produces")

    # Resource hints
    requires_auth: bool = False
    requires_network: bool = False
    is_stateful: bool = False
    typical_latency_ms: int = 1000
    cost_tier: str = Field(default="low", description="low/medium/high")

    # Usage hints
    best_for: list[str] = Field(default_factory=list, description="Scenarios this tool excels at")
    not_for: list[str] = Field(default_factory=list, description="Scenarios to avoid")

    # Relevance scoring helpers
    keywords: list[str] = Field(default_factory=list, description="Keywords for matching")
    example_queries: list[str] = Field(default_factory=list, description="Example queries this handles")


class ToolCandidate(BaseModel):
    """A tool candidate with relevance score."""

    metadata: ToolMetadata
    relevance_score: float = Field(ge=0.0, le=1.0, description="How relevant to the query")
    match_reasons: list[str] = Field(default_factory=list, description="Why this tool was selected")
    rank: int = Field(default=0, description="Rank in candidate list")


class ToolSelectionResult(BaseModel):
    """Result of tool selection."""

    candidates: list[ToolCandidate] = Field(default_factory=list)
    total_tools_available: int = 0
    selection_method: str = Field(default="keyword", description="How selection was done")
    query_categories: list[str] = Field(default_factory=list, description="Detected query categories")


class ToolSelector:
    """
    Efficient tool selector that filters to top candidates.

    Instead of sending all tool schemas to the LLM (expensive),
    this selector uses lightweight metadata to find the most
    relevant tools first.
    """

    def __init__(self, max_candidates: int = 10):
        self._metadata_registry: dict[str, ToolMetadata] = {}
        self._category_index: dict[str, list[str]] = {}
        self._keyword_index: dict[str, list[str]] = {}
        self._intent_index: dict[str, list[str]] = {}
        self.max_candidates = max_candidates

    def register_tool(self, metadata: ToolMetadata) -> None:
        """Register a tool's metadata for selection."""
        self._metadata_registry[metadata.tool_id] = metadata

        # Index by category
        if metadata.category not in self._category_index:
            self._category_index[metadata.category] = []
        self._category_index[metadata.category].append(metadata.tool_id)

        # Index by keywords
        for keyword in metadata.keywords + metadata.tags:
            keyword_lower = keyword.lower()
            if keyword_lower not in self._keyword_index:
                self._keyword_index[keyword_lower] = []
            self._keyword_index[keyword_lower].append(metadata.tool_id)

        # Index by intent
        for intent in metadata.supported_intents:
            if intent not in self._intent_index:
                self._intent_index[intent] = []
            self._intent_index[intent].append(metadata.tool_id)

    def unregister_tool(self, tool_id: str) -> None:
        """Remove a tool from the registry."""
        if tool_id in self._metadata_registry:
            del self._metadata_registry[tool_id]
            # Clean up indexes (simplified - in production would be more thorough)

    def select_candidates(
        self,
        query: str,
        query_understanding: QueryUnderstanding | None = None,
        categories: list[str] | None = None,
        max_candidates: int | None = None,
    ) -> ToolSelectionResult:
        """
        Select top candidate tools for a query.

        Uses multiple signals:
        1. Query understanding (intent, categories)
        2. Keyword matching
        3. Category filtering
        4. Metadata scoring

        Returns a lightweight result that can be expanded
        to full schemas only for selected tools.
        """
        max_candidates = max_candidates or self.max_candidates
        candidates: dict[str, float] = {}  # tool_id -> score
        match_reasons: dict[str, list[str]] = {}

        # 1. Intent-based matching
        if query_understanding:
            intent = query_understanding.intent.value
            if intent in self._intent_index:
                for tool_id in self._intent_index[intent]:
                    candidates[tool_id] = candidates.get(tool_id, 0) + 0.3
                    if tool_id not in match_reasons:
                        match_reasons[tool_id] = []
                    match_reasons[tool_id].append(f"matches intent: {intent}")

            # Suggested categories from understanding
            for category in query_understanding.suggested_tool_categories:
                if category in self._category_index:
                    for tool_id in self._category_index[category]:
                        candidates[tool_id] = candidates.get(tool_id, 0) + 0.25
                        if tool_id not in match_reasons:
                            match_reasons[tool_id] = []
                        match_reasons[tool_id].append(f"suggested category: {category}")

        # 2. Category filtering
        if categories:
            for category in categories:
                if category in self._category_index:
                    for tool_id in self._category_index[category]:
                        candidates[tool_id] = candidates.get(tool_id, 0) + 0.2
                        if tool_id not in match_reasons:
                            match_reasons[tool_id] = []
                        match_reasons[tool_id].append(f"category: {category}")

        # 3. Keyword matching
        query_words = set(query.lower().split())
        for word in query_words:
            if word in self._keyword_index:
                for tool_id in self._keyword_index[word]:
                    candidates[tool_id] = candidates.get(tool_id, 0) + 0.15
                    if tool_id not in match_reasons:
                        match_reasons[tool_id] = []
                    match_reasons[tool_id].append(f"keyword: {word}")

        # 4. Description matching (simple)
        for tool_id, metadata in self._metadata_registry.items():
            desc_words = set(metadata.description.lower().split())
            overlap = len(query_words & desc_words)
            if overlap > 0:
                candidates[tool_id] = candidates.get(tool_id, 0) + (0.1 * overlap)
                if tool_id not in match_reasons:
                    match_reasons[tool_id] = []
                match_reasons[tool_id].append(f"description overlap: {overlap} words")

        # 5. If no candidates, include some defaults
        if not candidates:
            # Add general-purpose tools
            for tool_id, metadata in self._metadata_registry.items():
                if "general" in metadata.tags or metadata.category == "general":
                    candidates[tool_id] = 0.1
                    match_reasons[tool_id] = ["default general tool"]

        # Sort by score and take top N
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        top_candidates = sorted_candidates[:max_candidates]

        # Build result
        result_candidates = []
        for rank, (tool_id, score) in enumerate(top_candidates, 1):
            metadata = self._metadata_registry.get(tool_id)
            if metadata:
                result_candidates.append(ToolCandidate(
                    metadata=metadata,
                    relevance_score=min(score, 1.0),
                    match_reasons=match_reasons.get(tool_id, []),
                    rank=rank,
                ))

        return ToolSelectionResult(
            candidates=result_candidates,
            total_tools_available=len(self._metadata_registry),
            selection_method="multi-signal",
            query_categories=categories or [],
        )

    def get_candidate_summary(self, result: ToolSelectionResult) -> str:
        """
        Get a compact summary of candidates for LLM.

        This is what gets sent to the LLM for tool selection,
        much smaller than full schemas.
        """
        if not result.candidates:
            return "No tools matched the query."

        lines = [f"Top {len(result.candidates)} tools (of {result.total_tools_available} available):"]

        for candidate in result.candidates:
            m = candidate.metadata
            score_pct = int(candidate.relevance_score * 100)
            lines.append(
                f"\n{candidate.rank}. **{m.name}** ({score_pct}% match)\n"
                f"   {m.description}\n"
                f"   Category: {m.category} | Best for: {', '.join(m.best_for[:3])}"
            )

        return "\n".join(lines)

    def get_all_categories(self) -> list[str]:
        """Get all registered tool categories."""
        return list(self._category_index.keys())

    def get_category_summary(self) -> str:
        """Get a summary of tool categories for LLM context."""
        lines = ["Available tool categories:"]
        for category, tool_ids in self._category_index.items():
            lines.append(f"- {category}: {len(tool_ids)} tools")
        return "\n".join(lines)


# Global tool selector instance
_tool_selector: ToolSelector | None = None


def get_tool_selector() -> ToolSelector:
    """Get or create the global tool selector."""
    global _tool_selector
    if _tool_selector is None:
        _tool_selector = ToolSelector()
    return _tool_selector
