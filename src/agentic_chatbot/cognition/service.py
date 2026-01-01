"""CognitionService: Unified interface for System 3 meta-cognitive layer.

This service provides a single interface for all cognitive operations:
- Context loading (fast, synchronous)
- Background learning (non-blocking via task queue)
- Component coordination (Theory of Mind, Episodic Memory, Identity)

Usage:
    # Initialize during app startup
    service = CognitionService()
    await service.initialize()

    # During request processing
    context = await service.get_context(user_id, query)

    # After response (non-blocking)
    await service.enqueue_learning(user_id, conversation_id, messages, outcome)

    # During shutdown
    await service.shutdown()
"""

from __future__ import annotations

import asyncio
from typing import Any

from agentic_chatbot.cognition.config import CognitionSettings, get_cognition_settings
from agentic_chatbot.cognition.models import (
    CognitiveContext,
    TaskType,
)
from agentic_chatbot.cognition.storage import CognitionStorage
from agentic_chatbot.cognition.task_queue import CognitionTaskQueue
from agentic_chatbot.cognition.theory_of_mind import TheoryOfMind
from agentic_chatbot.cognition.episodic_memory import EpisodicMemoryManager
from agentic_chatbot.cognition.identity import IdentityManager
from agentic_chatbot.cognition.meta_monitor import MetaMonitor
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class CognitionService:
    """
    Unified service for System 3 meta-cognitive operations.

    Coordinates all cognitive components and provides:
    - Fast context loading for request enrichment
    - Non-blocking background learning via task queue
    - Component lifecycle management
    """

    def __init__(self, settings: CognitionSettings | None = None):
        """
        Initialize cognition service.

        Args:
            settings: Optional settings (uses defaults if not provided)
        """
        self.settings = settings or get_cognition_settings()
        self._initialized = False

        # Components (initialized in initialize())
        self.storage: CognitionStorage | None = None
        self.task_queue: CognitionTaskQueue | None = None
        self.theory_of_mind: TheoryOfMind | None = None
        self.episodic_memory: EpisodicMemoryManager | None = None
        self.identity: IdentityManager | None = None
        self.meta_monitor: MetaMonitor | None = None

    async def initialize(self) -> None:
        """
        Initialize all cognitive components.

        Should be called during application startup.
        """
        if self._initialized:
            logger.warning("CognitionService already initialized")
            return

        if not self.settings.cognition_enabled:
            logger.info("Cognition service disabled")
            return

        logger.info("Initializing cognition service...")

        try:
            # Initialize storage
            self.storage = CognitionStorage(self.settings)
            await self.storage.initialize()

            # Initialize components
            self.theory_of_mind = TheoryOfMind(self.storage)
            self.episodic_memory = EpisodicMemoryManager(self.storage, self.settings)
            self.identity = IdentityManager(self.storage)
            self.meta_monitor = MetaMonitor(self.storage)

            # Initialize task queue
            self.task_queue = CognitionTaskQueue(self.storage, self.settings)
            self._register_task_handlers()
            await self.task_queue.start_worker()

            self._initialized = True
            logger.info("Cognition service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize cognition service: {e}", exc_info=True)
            # Clean up partial initialization
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        """
        Shutdown all cognitive components.

        Should be called during application shutdown.
        """
        logger.info("Shutting down cognition service...")

        # Stop task queue first
        if self.task_queue:
            await self.task_queue.stop_worker()

        # Close storage
        if self.storage:
            await self.storage.close()

        self._initialized = False
        logger.info("Cognition service shut down")

    def _register_task_handlers(self) -> None:
        """Register handlers for background tasks."""
        if not self.task_queue:
            return

        self.task_queue.register_handler(
            TaskType.LEARN_INTERACTION,
            self._handle_learn_interaction,
        )
        self.task_queue.register_handler(
            TaskType.UPDATE_PROFILE,
            self._handle_update_profile,
        )
        self.task_queue.register_handler(
            TaskType.PRUNE_MEMORIES,
            self._handle_prune_memories,
        )
        self.task_queue.register_handler(
            TaskType.UPDATE_IDENTITY,
            self._handle_update_identity,
        )

    # =========================================================================
    # PUBLIC INTERFACE
    # =========================================================================

    async def get_context(
        self,
        user_id: str,
        query: str,
    ) -> CognitiveContext:
        """
        Get cognitive context for request enrichment.

        This is a FAST, SYNCHRONOUS operation that should complete
        within the configured timeout (default: 100ms).

        Args:
            user_id: User identifier
            query: Current query

        Returns:
            CognitiveContext with user profile and relevant memories
        """
        if not self._initialized or not self.settings.cognition_enabled:
            return CognitiveContext()

        try:
            # Use timeout to ensure fast response
            timeout = self.settings.context_load_timeout_ms / 1000.0

            context = await asyncio.wait_for(
                self._load_context(user_id, query),
                timeout=timeout,
            )
            return context

        except asyncio.TimeoutError:
            logger.warning(f"Context loading timed out for user {user_id}")
            return CognitiveContext()
        except Exception as e:
            logger.error(f"Error loading context: {e}", exc_info=True)
            return CognitiveContext()

    async def enqueue_learning(
        self,
        user_id: str,
        conversation_id: str,
        messages: list[dict[str, Any]],
        outcome: str | None = None,
    ) -> str | None:
        """
        Enqueue a learning task for background processing.

        This is a FAST, NON-BLOCKING operation that returns immediately.
        The actual learning happens in the background worker.

        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            messages: Conversation messages
            outcome: Optional outcome (auto-analyzed if not provided)

        Returns:
            Task ID if enqueued, None if service disabled
        """
        if not self._initialized or not self.settings.cognition_enabled:
            return None

        if not self.task_queue:
            return None

        try:
            # Prepare payload
            payload = {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "messages": messages,
                "outcome": outcome,
            }

            # Enqueue learning task
            task_id = await self.task_queue.enqueue(
                TaskType.LEARN_INTERACTION,
                payload,
            )

            logger.debug(f"Enqueued learning task {task_id} for user {user_id}")
            return task_id

        except Exception as e:
            logger.error(f"Error enqueuing learning task: {e}", exc_info=True)
            return None

    async def get_user_profile(self, user_id: str) -> dict[str, Any] | None:
        """
        Get user profile for external access (e.g., API).

        Args:
            user_id: User identifier

        Returns:
            User profile dict or None if not found
        """
        if not self._initialized or not self.theory_of_mind:
            return None

        profile = await self.theory_of_mind.get_user_profile(user_id)
        return profile.to_dict()

    async def get_identity_summary(self) -> dict[str, Any] | None:
        """
        Get identity summary for external access.

        Returns:
            Identity summary dict or None if disabled
        """
        if not self._initialized or not self.identity:
            return None

        return await self.identity.get_performance_summary()

    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================

    async def _load_context(
        self,
        user_id: str,
        query: str,
    ) -> CognitiveContext:
        """Load cognitive context (internal, may be slow)."""
        context = CognitiveContext()

        # Load user profile
        if self.settings.theory_of_mind_enabled and self.theory_of_mind:
            context.user_profile = await self.theory_of_mind.get_user_profile(user_id)

        # Load relevant memories
        if self.settings.episodic_memory_enabled and self.episodic_memory:
            context.relevant_memories = await self.episodic_memory.get_relevant_memories(
                user_id=user_id,
                query=query,
            )

        # Load identity (minimal)
        if self.settings.identity_enabled and self.identity:
            context.identity = await self.identity.get_identity()

        return context

    # =========================================================================
    # TASK HANDLERS
    # =========================================================================

    async def _handle_learn_interaction(self, payload: dict[str, Any]) -> None:
        """Handle learn_interaction background task."""
        user_id = payload["user_id"]
        conversation_id = payload["conversation_id"]
        messages = payload["messages"]
        outcome = payload.get("outcome")

        # Analyze outcome if not provided
        if not outcome and self.identity:
            outcome = self.identity.analyze_outcome(messages, None)

        # Update user profile
        if self.settings.theory_of_mind_enabled and self.theory_of_mind:
            await self.theory_of_mind.update_from_interaction(
                user_id=user_id,
                messages=messages,
                outcome=outcome or "unknown",
            )

        # Create/merge memory
        if self.settings.episodic_memory_enabled and self.episodic_memory:
            summary = await self.episodic_memory.summarize_interaction(
                messages=messages,
                outcome=outcome or "unknown",
            )
            importance = self.episodic_memory.calculate_importance(
                messages=messages,
                outcome=outcome or "unknown",
            )
            await self.episodic_memory.create_or_merge_memory(
                user_id=user_id,
                conversation_id=conversation_id,
                summary=summary,
                outcome=outcome or "unknown",
                importance=importance,
            )

            # Prune if needed
            await self.episodic_memory.prune_memories(user_id)

        # Update identity
        if self.settings.identity_enabled and self.identity:
            await self.identity.record_interaction_outcome(
                outcome=outcome or "unknown",
            )

        logger.info(
            f"Learned from interaction for user {user_id}",
            outcome=outcome,
        )

    async def _handle_update_profile(self, payload: dict[str, Any]) -> None:
        """Handle update_profile background task."""
        user_id = payload["user_id"]
        updates = payload.get("updates", {})

        if not self.storage:
            return

        profile = await self.storage.get_or_create_user_profile(user_id)

        # Apply updates
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        await self.storage.update_user_profile(profile)
        logger.info(f"Updated profile for user {user_id}")

    async def _handle_prune_memories(self, payload: dict[str, Any]) -> None:
        """Handle prune_memories background task."""
        user_id = payload["user_id"]

        if self.episodic_memory:
            deleted = await self.episodic_memory.prune_memories(user_id)
            logger.info(f"Pruned {deleted} memories for user {user_id}")

    async def _handle_update_identity(self, payload: dict[str, Any]) -> None:
        """Handle update_identity background task."""
        goal = payload.get("goal")
        gap = payload.get("gap")
        value_name = payload.get("value_name")
        value_score = payload.get("value_score")

        if not self.identity:
            return

        if goal:
            await self.identity.add_learning_goal(goal)

        if gap:
            await self.identity.identify_knowledge_gap(gap)

        if value_name and value_score is not None:
            await self.identity.update_value(value_name, value_score)

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized

    @property
    def is_enabled(self) -> bool:
        """Check if service is enabled."""
        return self.settings.cognition_enabled
