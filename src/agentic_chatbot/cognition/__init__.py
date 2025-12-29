"""System 3: Meta-Cognitive Layer.

This module implements the meta-cognitive layer (System 3) that observes
and enriches the existing chat flow. It provides:

- Theory of Mind: User profiles and preference learning
- Episodic Memory: Cross-conversation persistent memory
- Identity: Learning goals and performance tracking
- Meta-Cognition: Self-reflection and confidence monitoring

Architecture:
    The cognition layer is non-blocking. Context enrichment happens
    synchronously at request start (fast), while learning happens
    asynchronously via a PostgreSQL-backed task queue after the
    response is sent.

Usage:
    from agentic_chatbot.cognition import CognitionService

    # Initialize with storage
    service = CognitionService(storage)

    # Get context for request enrichment
    context = await service.get_context(user_id, query)

    # Enqueue learning task (non-blocking)
    await service.enqueue_learning(user_id, conversation_id, messages, outcome)
"""

from agentic_chatbot.cognition.models import (
    UserProfile,
    EpisodicMemory,
    IdentityState,
    CognitiveContext,
    PerformanceMetric,
    LearningTask,
    TaskStatus,
)
from agentic_chatbot.cognition.config import CognitionSettings, get_cognition_settings
from agentic_chatbot.cognition.service import CognitionService

__all__ = [
    # Models
    "UserProfile",
    "EpisodicMemory",
    "IdentityState",
    "CognitiveContext",
    "PerformanceMetric",
    "LearningTask",
    "TaskStatus",
    # Config
    "CognitionSettings",
    "get_cognition_settings",
    # Service
    "CognitionService",
]
