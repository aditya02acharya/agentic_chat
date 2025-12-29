"""Data models for the meta-cognitive layer.

This module defines the core data structures for System 3 components:
- UserProfile: Theory of Mind user understanding
- EpisodicMemory: Cross-conversation persistent memory
- IdentityState: System's learning goals and values
- CognitiveContext: Injected context for request enrichment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ExpertiseLevel(str, Enum):
    """User expertise level."""

    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


class CommunicationStyle(str, Enum):
    """User communication preference."""

    CONCISE = "concise"
    DETAILED = "detailed"
    TECHNICAL = "technical"


class TaskStatus(str, Enum):
    """Background task status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(str, Enum):
    """Types of background cognition tasks."""

    LEARN_INTERACTION = "learn_interaction"
    UPDATE_PROFILE = "update_profile"
    PRUNE_MEMORIES = "prune_memories"
    UPDATE_IDENTITY = "update_identity"


# =============================================================================
# USER PROFILE (THEORY OF MIND)
# =============================================================================


@dataclass
class UserProfile:
    """
    Theory of Mind - User understanding.

    Tracks user expertise, preferences, and interaction patterns
    to personalize responses and anticipate needs.
    """

    user_id: str
    expertise_level: ExpertiseLevel = ExpertiseLevel.INTERMEDIATE
    communication_style: CommunicationStyle = CommunicationStyle.DETAILED
    domain_interests: list[str] = field(default_factory=list)
    preferences: dict[str, Any] = field(default_factory=dict)
    interaction_count: int = 0
    successful_interactions: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "expertise_level": self.expertise_level.value,
            "communication_style": self.communication_style.value,
            "domain_interests": self.domain_interests,
            "preferences": self.preferences,
            "interaction_count": self.interaction_count,
            "successful_interactions": self.successful_interactions,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserProfile:
        """Deserialize from dictionary."""
        return cls(
            user_id=data["user_id"],
            expertise_level=ExpertiseLevel(data.get("expertise_level", "intermediate")),
            communication_style=CommunicationStyle(data.get("communication_style", "detailed")),
            domain_interests=data.get("domain_interests", []),
            preferences=data.get("preferences", {}),
            interaction_count=data.get("interaction_count", 0),
            successful_interactions=data.get("successful_interactions", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
        )

    def to_context_text(self) -> str:
        """Format for supervisor context injection."""
        lines = [
            f"**User Profile** (ID: {self.user_id})",
            f"- Expertise: {self.expertise_level.value}",
            f"- Communication style: {self.communication_style.value}",
        ]
        if self.domain_interests:
            lines.append(f"- Interests: {', '.join(self.domain_interests[:5])}")
        if self.preferences.get("prefers_examples"):
            lines.append("- Prefers examples in explanations")
        if self.preferences.get("prefers_code"):
            lines.append("- Prefers code snippets")
        lines.append(f"- Interactions: {self.interaction_count} ({self.successful_interactions} successful)")
        return "\n".join(lines)


# =============================================================================
# EPISODIC MEMORY
# =============================================================================


@dataclass
class EpisodicMemory:
    """
    Cross-conversation memory entry.

    Stores summaries of past interactions with importance scoring
    for retrieval and pruning decisions.
    """

    memory_id: str
    user_id: str
    conversation_id: str
    summary: str  # What happened in this interaction
    outcome: str  # success | partial | failure
    topics: list[str] = field(default_factory=list)
    importance: float = 0.5  # 0-1 scale
    access_count: int = 0
    merged_count: int = 1  # How many memories consolidated into this
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "summary": self.summary,
            "outcome": self.outcome,
            "topics": self.topics,
            "importance": self.importance,
            "access_count": self.access_count,
            "merged_count": self.merged_count,
            "last_accessed": self.last_accessed.isoformat(),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodicMemory:
        """Deserialize from dictionary."""
        return cls(
            memory_id=data["memory_id"],
            user_id=data["user_id"],
            conversation_id=data["conversation_id"],
            summary=data["summary"],
            outcome=data.get("outcome", "unknown"),
            topics=data.get("topics", []),
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            merged_count=data.get("merged_count", 1),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else datetime.utcnow(),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
        )

    def to_context_text(self) -> str:
        """Format for supervisor context injection."""
        topics_str = ", ".join(self.topics[:3]) if self.topics else "general"
        return f"- [{self.outcome}] {self.summary} (topics: {topics_str})"

    def relevance_score(self, query_topics: list[str]) -> float:
        """
        Calculate relevance to a query based on topic overlap.

        Args:
            query_topics: Topics extracted from the current query

        Returns:
            Relevance score between 0 and 1
        """
        if not query_topics or not self.topics:
            return 0.0

        query_set = set(t.lower() for t in query_topics)
        memory_set = set(t.lower() for t in self.topics)
        overlap = len(query_set & memory_set)
        union = len(query_set | memory_set)

        return overlap / union if union > 0 else 0.0


# =============================================================================
# IDENTITY STATE
# =============================================================================


@dataclass
class PerformanceMetric:
    """A single performance metric entry."""

    metric_name: str
    value: float
    recorded_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "recorded_at": self.recorded_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerformanceMetric:
        """Deserialize from dictionary."""
        return cls(
            metric_name=data["metric_name"],
            value=data["value"],
            recorded_at=datetime.fromisoformat(data["recorded_at"]) if data.get("recorded_at") else datetime.utcnow(),
        )


@dataclass
class IdentityState:
    """
    Persistent identity and goals.

    Tracks the system's learning objectives, identified knowledge gaps,
    and accumulated performance metrics.
    """

    learning_goals: list[str] = field(default_factory=list)
    knowledge_gaps: list[str] = field(default_factory=list)
    performance_history: list[PerformanceMetric] = field(default_factory=list)
    values: dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.9,
        "helpfulness": 0.95,
        "clarity": 0.9,
    })
    total_interactions: int = 0
    successful_interactions: int = 0
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "learning_goals": self.learning_goals,
            "knowledge_gaps": self.knowledge_gaps,
            "performance_history": [m.to_dict() for m in self.performance_history[-100:]],  # Keep last 100
            "values": self.values,
            "total_interactions": self.total_interactions,
            "successful_interactions": self.successful_interactions,
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IdentityState:
        """Deserialize from dictionary."""
        return cls(
            learning_goals=data.get("learning_goals", []),
            knowledge_gaps=data.get("knowledge_gaps", []),
            performance_history=[PerformanceMetric.from_dict(m) for m in data.get("performance_history", [])],
            values=data.get("values", {"accuracy": 0.9, "helpfulness": 0.95, "clarity": 0.9}),
            total_interactions=data.get("total_interactions", 0),
            successful_interactions=data.get("successful_interactions", 0),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
        )

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_interactions == 0:
            return 0.0
        return self.successful_interactions / self.total_interactions

    def to_context_text(self) -> str:
        """Format for supervisor context (minimal)."""
        lines = ["**System Identity**"]
        if self.learning_goals:
            lines.append(f"- Current goals: {', '.join(self.learning_goals[:3])}")
        if self.knowledge_gaps:
            lines.append(f"- Known gaps: {', '.join(self.knowledge_gaps[:3])}")
        lines.append(f"- Success rate: {self.success_rate:.1%}")
        return "\n".join(lines)


# =============================================================================
# COGNITIVE CONTEXT
# =============================================================================


@dataclass
class CognitiveContext:
    """
    Injected context for request enrichment.

    This is loaded synchronously at request start and provides
    the supervisor with user understanding and relevant memories.
    """

    user_profile: UserProfile | None = None
    relevant_memories: list[EpisodicMemory] = field(default_factory=list)
    identity: IdentityState | None = None

    def to_supervisor_text(self) -> str:
        """Format all context for supervisor prompt injection."""
        sections = []

        if self.user_profile:
            sections.append(self.user_profile.to_context_text())

        if self.relevant_memories:
            memory_lines = ["**Relevant Past Interactions**"]
            for mem in self.relevant_memories[:5]:  # Top 5 memories
                memory_lines.append(mem.to_context_text())
            sections.append("\n".join(memory_lines))

        if self.identity:
            sections.append(self.identity.to_context_text())

        return "\n\n".join(sections) if sections else ""

    def has_context(self) -> bool:
        """Check if any context is available."""
        return bool(self.user_profile or self.relevant_memories or self.identity)


# =============================================================================
# BACKGROUND TASK
# =============================================================================


@dataclass
class LearningTask:
    """
    Background learning task for the task queue.

    Tasks are stored in PostgreSQL and processed by a single worker
    with retry support and exponential backoff.
    """

    task_id: str
    task_type: TaskType
    payload: dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_for: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "payload": self.payload,
            "status": self.status.value,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "created_at": self.created_at.isoformat(),
            "scheduled_for": self.scheduled_for.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearningTask:
        """Deserialize from dictionary."""
        return cls(
            task_id=data["task_id"],
            task_type=TaskType(data["task_type"]),
            payload=data["payload"],
            status=TaskStatus(data.get("status", "pending")),
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", 3),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            scheduled_for=datetime.fromisoformat(data["scheduled_for"]) if data.get("scheduled_for") else datetime.utcnow(),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            error=data.get("error"),
        )
