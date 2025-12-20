"""Context management models for optimized information flow.

DEPRECATED: This module contains legacy data structures.
Use the unified data model from agentic_chatbot.data instead:
- TaskContext -> TaskInfo (from data.execution)
- DataChunk + DataSummary -> SourcedContent (from data.sourced)
- DataStore -> use state.sourced_contents directly

These structures are kept for backward compatibility during migration.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_chatbot.data.execution import TaskInfo
    from agentic_chatbot.data.sourced import SourcedContent, ContentSummary


@dataclass
class TaskContext:
    """
    Focused context for operators - what the supervisor delegates.

    DEPRECATED: Use TaskInfo from agentic_chatbot.data.execution instead.

    Operators receive this instead of full conversation history,
    keeping their context focused on the specific task.
    """

    task_description: str  # Reformulated task from supervisor
    goal: str  # Expected outcome
    scope: str = ""  # What's in/out of scope
    constraints: list[str] = field(default_factory=list)  # Any limits
    hints: dict[str, Any] = field(default_factory=dict)  # Additional guidance

    # Reference to original query (for grounding, not full context)
    original_query_summary: str = ""

    def to_prompt(self) -> str:
        """Format as prompt section for operators."""
        parts = [
            f"## Task\n{self.task_description}",
            f"\n## Goal\n{self.goal}",
        ]

        if self.scope:
            parts.append(f"\n## Scope\n{self.scope}")

        if self.constraints:
            constraints_text = "\n".join(f"- {c}" for c in self.constraints)
            parts.append(f"\n## Constraints\n{constraints_text}")

        if self.original_query_summary:
            parts.append(f"\n## Context\nUser's original request: {self.original_query_summary}")

        return "\n".join(parts)

    def to_task_info(self) -> "TaskInfo":
        """Convert to new TaskInfo type."""
        from agentic_chatbot.data.execution import TaskInfo

        return TaskInfo(
            description=self.task_description,
            goal=self.goal,
            scope=self.scope,
            constraints=self.constraints,
            hints=self.hints,
            original_query=self.original_query_summary,
        )


@dataclass
class DataChunk:
    """
    Raw data with source tracking for citations.

    DEPRECATED: Use SourcedContent from agentic_chatbot.data.sourced instead.
    SourcedContent combines both the raw content and its summary.

    Stored verbatim for synthesizer/writer to use with proper citations.
    Each chunk gets a unique source_id for footnote references.
    """

    source_id: str  # Unique ID for citation, e.g., "search_1", "rag_2"
    source_type: str  # Tool/operator name, e.g., "web_search", "rag_search"
    content: str  # Verbatim content from the tool

    # Metadata for context
    query_used: str = ""  # What query was used to get this data
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    # For ordering/relevance (optional, set by tool if available)
    relevance_hint: float | None = None  # 0-1 if tool provides it

    def to_citation_block(self) -> str:
        """Format as citation block for writer."""
        return f"[^{self.source_id}]: Source: {self.source_type}\n{self.content}"

    def get_footnote_ref(self) -> str:
        """Get the footnote reference marker."""
        return f"[^{self.source_id}]"

    def to_sourced_content(self, summary: "DataSummary | None" = None) -> "SourcedContent":
        """Convert to new SourcedContent type."""
        from agentic_chatbot.data.sourced import (
            SourcedContent,
            ContentSource,
            ContentSummary,
            create_sourced_content,
        )

        sourced = create_sourced_content(
            content=self.content,
            source_type=self.source_type,
            source_id=self.source_id,
            query_used=self.query_used,
        )

        if summary:
            sourced.set_summary(ContentSummary(
                executive_summary=summary.executive_summary,
                key_findings=summary.key_findings,
                has_useful_data=summary.has_results,
                error=summary.error,
                task_description=summary.task_description,
            ))

        return sourced


@dataclass
class DataSummary:
    """
    Condensed summary for supervisor decision-making.

    DEPRECATED: Use ContentSummary from agentic_chatbot.data.sourced instead.
    ContentSummary is part of SourcedContent, which unifies chunk + summary.

    Generated via LLM (haiku for speed) right after tool execution.
    Supervisor uses these to decide next action without seeing raw data.
    """

    source_id: str  # Links to corresponding DataChunk
    source_type: str  # Tool/operator name

    # LLM-generated summary
    key_findings: list[str]  # Main points (bullet form)
    executive_summary: str  # One-line summary

    # Simple status (not LLM-scored, just factual)
    has_results: bool = True  # Did the tool return data?
    error: str | None = None  # Any error message

    # For supervisor context
    task_description: str = ""  # What task was this for

    def to_supervisor_text(self) -> str:
        """Format for supervisor's context window."""
        if self.error:
            return f"**{self.source_type}** [{self.source_id}]: ❌ Error - {self.error}"

        if not self.has_results:
            return f"**{self.source_type}** [{self.source_id}]: No results found"

        findings = "\n".join(f"  - {f}" for f in self.key_findings[:3])  # Limit to top 3
        return f"**{self.source_type}** [{self.source_id}]: {self.executive_summary}\n{findings}"

    def to_content_summary(self) -> "ContentSummary":
        """Convert to new ContentSummary type."""
        from agentic_chatbot.data.sourced import ContentSummary

        return ContentSummary(
            executive_summary=self.executive_summary,
            key_findings=self.key_findings,
            has_useful_data=self.has_results,
            error=self.error,
            task_description=self.task_description,
        )


@dataclass
class DataStore:
    """
    Container for all data collected during a conversation turn.

    DEPRECATED: Use state.sourced_contents directly instead.
    Each SourcedContent contains both the raw content and its summary.

    Maintains both raw chunks (for synthesizer/writer) and
    summaries (for supervisor).
    """

    chunks: list[DataChunk] = field(default_factory=list)
    summaries: list[DataSummary] = field(default_factory=list)

    _source_counter: dict[str, int] = field(default_factory=dict)

    def generate_source_id(self, source_type: str) -> str:
        """Generate unique source ID for a new chunk."""
        count = self._source_counter.get(source_type, 0) + 1
        self._source_counter[source_type] = count
        return f"{source_type}_{count}"

    def add_chunk(self, chunk: DataChunk) -> None:
        """Add a data chunk."""
        self.chunks.append(chunk)

    def add_summary(self, summary: DataSummary) -> None:
        """Add a summary."""
        self.summaries.append(summary)

    def get_summaries_for_supervisor(self) -> str:
        """Get formatted summaries for supervisor context."""
        if not self.summaries:
            return "No data collected yet."

        return "\n\n".join(s.to_supervisor_text() for s in self.summaries)

    def get_chunks_for_synthesis(self) -> list[DataChunk]:
        """Get all chunks for synthesizer/writer."""
        return self.chunks

    def get_citation_blocks(self) -> str:
        """Get all citation blocks for footnotes."""
        return "\n\n".join(c.to_citation_block() for c in self.chunks)

    def get_chunk_by_id(self, source_id: str) -> DataChunk | None:
        """Get specific chunk by source ID."""
        for chunk in self.chunks:
            if chunk.source_id == source_id:
                return chunk
        return None

    def to_sourced_contents(self) -> list["SourcedContent"]:
        """Convert all chunks to SourcedContent list."""
        # Match chunks with their summaries
        summary_map = {s.source_id: s for s in self.summaries}
        return [
            chunk.to_sourced_content(summary_map.get(chunk.source_id))
            for chunk in self.chunks
        ]
