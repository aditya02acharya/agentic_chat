"""PostgreSQL storage layer for the meta-cognitive system.

Handles all database operations for:
- User profiles (Theory of Mind)
- Episodic memories (Cross-conversation memory)
- Identity state (Learning goals and metrics)
- Background tasks (Task queue)

Uses asyncpg for async PostgreSQL operations.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import asyncpg

from agentic_chatbot.cognition.config import CognitionSettings, get_cognition_settings
from agentic_chatbot.cognition.models import (
    UserProfile,
    EpisodicMemory,
    IdentityState,
    LearningTask,
    TaskStatus,
    TaskType,
)
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


# =============================================================================
# SQL SCHEMA
# =============================================================================

SCHEMA_SQL = """
-- User profiles for Theory of Mind
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    expertise_level TEXT DEFAULT 'intermediate',
    communication_style TEXT DEFAULT 'detailed',
    domain_interests JSONB DEFAULT '[]',
    preferences JSONB DEFAULT '{}',
    interaction_count INTEGER DEFAULT 0,
    successful_interactions INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Episodic memories for cross-conversation persistence
CREATE TABLE IF NOT EXISTS episodic_memories (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    outcome TEXT DEFAULT 'unknown',
    topics JSONB DEFAULT '[]',
    importance REAL DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    merged_count INTEGER DEFAULT 1,
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Identity state (singleton table)
CREATE TABLE IF NOT EXISTS identity_state (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    learning_goals JSONB DEFAULT '[]',
    knowledge_gaps JSONB DEFAULT '[]',
    performance_history JSONB DEFAULT '[]',
    values JSONB DEFAULT '{"accuracy": 0.9, "helpfulness": 0.95, "clarity": 0.9}',
    total_interactions INTEGER DEFAULT 0,
    successful_interactions INTEGER DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Background task queue
CREATE TABLE IF NOT EXISTS cognition_tasks (
    task_id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    payload JSONB NOT NULL,
    status TEXT DEFAULT 'pending',
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    scheduled_for TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error TEXT
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_memories_user_importance
    ON episodic_memories(user_id, importance DESC);
CREATE INDEX IF NOT EXISTS idx_memories_user_recency
    ON episodic_memories(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_topics
    ON episodic_memories USING GIN(topics);
CREATE INDEX IF NOT EXISTS idx_tasks_pending
    ON cognition_tasks(status, scheduled_for)
    WHERE status = 'pending';

-- Initialize identity state if not exists
INSERT INTO identity_state (id) VALUES (1) ON CONFLICT (id) DO NOTHING;
"""


class CognitionStorage:
    """
    PostgreSQL storage for the meta-cognitive system.

    Provides async operations for all cognition data:
    - User profiles
    - Episodic memories
    - Identity state
    - Background tasks
    """

    def __init__(self, settings: CognitionSettings | None = None):
        """
        Initialize storage.

        Args:
            settings: Cognition settings (uses defaults if not provided)
        """
        self.settings = settings or get_cognition_settings()
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        """
        Initialize database connection pool and schema.

        Should be called during application startup.
        """
        logger.info("Initializing cognition storage...")

        # Create connection pool
        self._pool = await asyncpg.create_pool(
            host=self.settings.cognition_db_host,
            port=self.settings.cognition_db_port,
            database=self.settings.cognition_db_name,
            user=self.settings.cognition_db_user,
            password=self.settings.cognition_db_password,
            min_size=2,
            max_size=self.settings.cognition_db_pool_size,
        )

        # Create schema
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)

        logger.info("Cognition storage initialized")

    async def close(self) -> None:
        """Close database connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Cognition storage closed")

    @property
    def pool(self) -> asyncpg.Pool:
        """Get connection pool."""
        if not self._pool:
            raise RuntimeError("Storage not initialized. Call initialize() first.")
        return self._pool

    # =========================================================================
    # USER PROFILE OPERATIONS
    # =========================================================================

    async def get_user_profile(self, user_id: str) -> UserProfile | None:
        """Get user profile by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM user_profiles WHERE user_id = $1",
                user_id,
            )
            if row:
                return self._row_to_user_profile(row)
            return None

    async def create_user_profile(self, profile: UserProfile) -> None:
        """Create a new user profile."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO user_profiles (
                    user_id, expertise_level, communication_style,
                    domain_interests, preferences, interaction_count,
                    successful_interactions, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                profile.user_id,
                profile.expertise_level.value,
                profile.communication_style.value,
                json.dumps(profile.domain_interests),
                json.dumps(profile.preferences),
                profile.interaction_count,
                profile.successful_interactions,
                profile.created_at,
                profile.updated_at,
            )

    async def update_user_profile(self, profile: UserProfile) -> None:
        """Update an existing user profile."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE user_profiles SET
                    expertise_level = $2,
                    communication_style = $3,
                    domain_interests = $4,
                    preferences = $5,
                    interaction_count = $6,
                    successful_interactions = $7,
                    updated_at = $8
                WHERE user_id = $1
                """,
                profile.user_id,
                profile.expertise_level.value,
                profile.communication_style.value,
                json.dumps(profile.domain_interests),
                json.dumps(profile.preferences),
                profile.interaction_count,
                profile.successful_interactions,
                datetime.utcnow(),
            )

    async def get_or_create_user_profile(self, user_id: str) -> UserProfile:
        """Get existing profile or create a new one."""
        profile = await self.get_user_profile(user_id)
        if profile:
            return profile

        # Create new profile
        profile = UserProfile(user_id=user_id)
        await self.create_user_profile(profile)
        return profile

    async def increment_user_interaction(
        self, user_id: str, successful: bool = True
    ) -> None:
        """Increment user interaction count."""
        async with self.pool.acquire() as conn:
            if successful:
                await conn.execute(
                    """
                    UPDATE user_profiles SET
                        interaction_count = interaction_count + 1,
                        successful_interactions = successful_interactions + 1,
                        updated_at = NOW()
                    WHERE user_id = $1
                    """,
                    user_id,
                )
            else:
                await conn.execute(
                    """
                    UPDATE user_profiles SET
                        interaction_count = interaction_count + 1,
                        updated_at = NOW()
                    WHERE user_id = $1
                    """,
                    user_id,
                )

    def _row_to_user_profile(self, row: asyncpg.Record) -> UserProfile:
        """Convert database row to UserProfile."""
        from agentic_chatbot.cognition.models import ExpertiseLevel, CommunicationStyle

        return UserProfile(
            user_id=row["user_id"],
            expertise_level=ExpertiseLevel(row["expertise_level"]),
            communication_style=CommunicationStyle(row["communication_style"]),
            domain_interests=json.loads(row["domain_interests"]) if isinstance(row["domain_interests"], str) else row["domain_interests"],
            preferences=json.loads(row["preferences"]) if isinstance(row["preferences"], str) else row["preferences"],
            interaction_count=row["interaction_count"],
            successful_interactions=row["successful_interactions"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # =========================================================================
    # EPISODIC MEMORY OPERATIONS
    # =========================================================================

    async def get_memories_for_user(
        self,
        user_id: str,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> list[EpisodicMemory]:
        """Get memories for a user ordered by importance and recency."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM episodic_memories
                WHERE user_id = $1 AND importance >= $2
                ORDER BY importance DESC, created_at DESC
                LIMIT $3
                """,
                user_id,
                min_importance,
                limit,
            )
            return [self._row_to_memory(row) for row in rows]

    async def get_memories_by_topics(
        self,
        user_id: str,
        topics: list[str],
        limit: int = 5,
    ) -> list[EpisodicMemory]:
        """Get memories that match given topics."""
        if not topics:
            return []

        async with self.pool.acquire() as conn:
            # Use JSONB containment to find overlapping topics
            rows = await conn.fetch(
                """
                SELECT *,
                    (SELECT COUNT(*) FROM jsonb_array_elements_text(topics) t
                     WHERE t = ANY($2::text[])) as topic_overlap
                FROM episodic_memories
                WHERE user_id = $1
                    AND topics ?| $2::text[]
                ORDER BY topic_overlap DESC, importance DESC
                LIMIT $3
                """,
                user_id,
                topics,
                limit,
            )
            return [self._row_to_memory(row) for row in rows]

    async def create_memory(self, memory: EpisodicMemory) -> None:
        """Create a new episodic memory."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO episodic_memories (
                    memory_id, user_id, conversation_id, summary, outcome,
                    topics, importance, access_count, merged_count,
                    last_accessed, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                memory.memory_id,
                memory.user_id,
                memory.conversation_id,
                memory.summary,
                memory.outcome,
                json.dumps(memory.topics),
                memory.importance,
                memory.access_count,
                memory.merged_count,
                memory.last_accessed,
                memory.created_at,
            )

    async def update_memory(
        self,
        memory_id: str,
        summary: str | None = None,
        topics: list[str] | None = None,
        importance: float | None = None,
        merged_count: int | None = None,
    ) -> None:
        """Update an existing memory (for merging)."""
        updates = []
        params = [memory_id]
        param_idx = 2

        if summary is not None:
            updates.append(f"summary = ${param_idx}")
            params.append(summary)
            param_idx += 1

        if topics is not None:
            updates.append(f"topics = ${param_idx}")
            params.append(json.dumps(topics))
            param_idx += 1

        if importance is not None:
            updates.append(f"importance = ${param_idx}")
            params.append(importance)
            param_idx += 1

        if merged_count is not None:
            updates.append(f"merged_count = ${param_idx}")
            params.append(merged_count)
            param_idx += 1

        if not updates:
            return

        updates.append("last_accessed = NOW()")

        async with self.pool.acquire() as conn:
            await conn.execute(
                f"UPDATE episodic_memories SET {', '.join(updates)} WHERE memory_id = $1",
                *params,
            )

    async def increment_memory_access(self, memory_id: str) -> None:
        """Increment access count and update last_accessed."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE episodic_memories SET
                    access_count = access_count + 1,
                    last_accessed = NOW()
                WHERE memory_id = $1
                """,
                memory_id,
            )

    async def find_similar_memories(
        self,
        user_id: str,
        topics: list[str],
        days: int = 7,
        limit: int = 5,
    ) -> list[EpisodicMemory]:
        """Find similar memories for deduplication."""
        if not topics:
            return []

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM episodic_memories
                WHERE user_id = $1
                    AND topics ?| $2::text[]
                    AND created_at > NOW() - INTERVAL '%s days'
                ORDER BY created_at DESC
                LIMIT $3
                """ % days,
                user_id,
                topics,
                limit,
            )
            return [self._row_to_memory(row) for row in rows]

    async def count_memories(self, user_id: str) -> int:
        """Count total memories for a user."""
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT COUNT(*) FROM episodic_memories WHERE user_id = $1",
                user_id,
            )
            return result or 0

    async def prune_old_memories(
        self,
        user_id: str,
        max_memories: int,
        ttl_days: int,
        low_importance_threshold: float,
    ) -> int:
        """
        Prune old, low-importance memories.

        Returns the number of memories deleted.
        """
        async with self.pool.acquire() as conn:
            # First, delete old low-importance memories
            result1 = await conn.execute(
                """
                DELETE FROM episodic_memories
                WHERE user_id = $1
                    AND importance < $2
                    AND created_at < NOW() - INTERVAL '%s days'
                """ % ttl_days,
                user_id,
                low_importance_threshold,
            )
            deleted1 = int(result1.split()[-1]) if result1 else 0

            # Check if still over limit
            count = await self.count_memories(user_id)
            if count <= max_memories:
                return deleted1

            # Delete excess memories by composite score
            excess = count - max_memories
            result2 = await conn.execute(
                """
                DELETE FROM episodic_memories
                WHERE memory_id IN (
                    SELECT memory_id FROM episodic_memories
                    WHERE user_id = $1
                    ORDER BY
                        (importance * 0.4) +
                        (merged_count * 0.1) +
                        (access_count * 0.3) +
                        (1.0 / (EXTRACT(EPOCH FROM NOW() - last_accessed) / 86400 + 1) * 0.2)
                    ASC
                    LIMIT $2
                )
                """,
                user_id,
                excess,
            )
            deleted2 = int(result2.split()[-1]) if result2 else 0

            return deleted1 + deleted2

    def _row_to_memory(self, row: asyncpg.Record) -> EpisodicMemory:
        """Convert database row to EpisodicMemory."""
        return EpisodicMemory(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            conversation_id=row["conversation_id"],
            summary=row["summary"],
            outcome=row["outcome"],
            topics=json.loads(row["topics"]) if isinstance(row["topics"], str) else row["topics"],
            importance=row["importance"],
            access_count=row["access_count"],
            merged_count=row["merged_count"],
            last_accessed=row["last_accessed"],
            created_at=row["created_at"],
        )

    # =========================================================================
    # IDENTITY STATE OPERATIONS
    # =========================================================================

    async def get_identity_state(self) -> IdentityState:
        """Get the singleton identity state."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM identity_state WHERE id = 1")
            if row:
                return self._row_to_identity(row)
            # Should not happen due to schema initialization, but handle it
            return IdentityState()

    async def update_identity_state(self, identity: IdentityState) -> None:
        """Update the identity state."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE identity_state SET
                    learning_goals = $1,
                    knowledge_gaps = $2,
                    performance_history = $3,
                    values = $4,
                    total_interactions = $5,
                    successful_interactions = $6,
                    updated_at = NOW()
                WHERE id = 1
                """,
                json.dumps(identity.learning_goals),
                json.dumps(identity.knowledge_gaps),
                json.dumps([m.to_dict() for m in identity.performance_history[-100:]]),
                json.dumps(identity.values),
                identity.total_interactions,
                identity.successful_interactions,
            )

    async def increment_identity_interaction(self, successful: bool = True) -> None:
        """Increment identity interaction count."""
        async with self.pool.acquire() as conn:
            if successful:
                await conn.execute(
                    """
                    UPDATE identity_state SET
                        total_interactions = total_interactions + 1,
                        successful_interactions = successful_interactions + 1,
                        updated_at = NOW()
                    WHERE id = 1
                    """
                )
            else:
                await conn.execute(
                    """
                    UPDATE identity_state SET
                        total_interactions = total_interactions + 1,
                        updated_at = NOW()
                    WHERE id = 1
                    """
                )

    def _row_to_identity(self, row: asyncpg.Record) -> IdentityState:
        """Convert database row to IdentityState."""
        from agentic_chatbot.cognition.models import PerformanceMetric

        performance_data = row["performance_history"]
        if isinstance(performance_data, str):
            performance_data = json.loads(performance_data)

        return IdentityState(
            learning_goals=json.loads(row["learning_goals"]) if isinstance(row["learning_goals"], str) else row["learning_goals"],
            knowledge_gaps=json.loads(row["knowledge_gaps"]) if isinstance(row["knowledge_gaps"], str) else row["knowledge_gaps"],
            performance_history=[PerformanceMetric.from_dict(m) for m in performance_data],
            values=json.loads(row["values"]) if isinstance(row["values"], str) else row["values"],
            total_interactions=row["total_interactions"],
            successful_interactions=row["successful_interactions"],
            updated_at=row["updated_at"],
        )

    # =========================================================================
    # TASK QUEUE OPERATIONS
    # =========================================================================

    async def enqueue_task(self, task: LearningTask) -> str:
        """Add a task to the queue."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO cognition_tasks (
                    task_id, task_type, payload, status, attempts,
                    max_attempts, created_at, scheduled_for
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                task.task_id,
                task.task_type.value,
                json.dumps(task.payload),
                task.status.value,
                task.attempts,
                task.max_attempts,
                task.created_at,
                task.scheduled_for,
            )
        return task.task_id

    async def claim_next_task(self) -> LearningTask | None:
        """
        Atomically claim the next pending task.

        Uses FOR UPDATE SKIP LOCKED to prevent race conditions
        when multiple workers are running (future-proofing).
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE cognition_tasks
                SET status = 'processing',
                    started_at = NOW(),
                    attempts = attempts + 1
                WHERE task_id = (
                    SELECT task_id FROM cognition_tasks
                    WHERE status = 'pending'
                        AND scheduled_for <= NOW()
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                RETURNING *
                """
            )
            if row:
                return self._row_to_task(row)
            return None

    async def complete_task(self, task_id: str) -> None:
        """Mark a task as completed."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE cognition_tasks
                SET status = 'completed', completed_at = NOW()
                WHERE task_id = $1
                """,
                task_id,
            )

    async def fail_task(self, task_id: str, error: str, retry: bool = True) -> None:
        """Mark a task as failed, optionally scheduling retry."""
        async with self.pool.acquire() as conn:
            if retry:
                # Get current attempts
                row = await conn.fetchrow(
                    "SELECT attempts, max_attempts FROM cognition_tasks WHERE task_id = $1",
                    task_id,
                )
                if row and row["attempts"] < row["max_attempts"]:
                    # Schedule retry with exponential backoff
                    backoff = 2 ** row["attempts"]
                    await conn.execute(
                        """
                        UPDATE cognition_tasks
                        SET status = 'pending',
                            scheduled_for = NOW() + INTERVAL '%s seconds',
                            error = $2
                        WHERE task_id = $1
                        """ % backoff,
                        task_id,
                        error,
                    )
                    return

            # Max retries exceeded or no retry requested
            await conn.execute(
                """
                UPDATE cognition_tasks
                SET status = 'failed', error = $2, completed_at = NOW()
                WHERE task_id = $1
                """,
                task_id,
                error,
            )

    async def cleanup_old_tasks(self, days: int = 7) -> int:
        """Delete completed/failed tasks older than specified days."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM cognition_tasks
                WHERE status IN ('completed', 'failed')
                    AND completed_at < NOW() - INTERVAL '%s days'
                """ % days
            )
            return int(result.split()[-1]) if result else 0

    def _row_to_task(self, row: asyncpg.Record) -> LearningTask:
        """Convert database row to LearningTask."""
        return LearningTask(
            task_id=row["task_id"],
            task_type=TaskType(row["task_type"]),
            payload=json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"],
            status=TaskStatus(row["status"]),
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            created_at=row["created_at"],
            scheduled_for=row["scheduled_for"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            error=row["error"],
        )
