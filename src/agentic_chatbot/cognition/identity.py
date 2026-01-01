"""Identity: Persistent goals, values, and performance tracking.

This component maintains the system's persistent identity:
- Learning goals (areas to improve)
- Knowledge gaps (identified weaknesses)
- Performance metrics (success rates, patterns)
- Core values (accuracy, helpfulness, clarity)

The identity evolves over time based on interaction outcomes,
providing intrinsic motivation for self-improvement.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from agentic_chatbot.cognition.models import IdentityState, PerformanceMetric
from agentic_chatbot.utils.logging import get_logger

if TYPE_CHECKING:
    from agentic_chatbot.cognition.storage import CognitionStorage


logger = get_logger(__name__)


# Default learning goals for a new system
DEFAULT_LEARNING_GOALS = [
    "Improve response accuracy",
    "Better understand user intent",
    "Provide more helpful examples",
]


class IdentityManager:
    """
    Manager for the system's persistent identity.

    Tracks learning objectives, knowledge gaps, and performance
    to guide self-improvement over time.
    """

    def __init__(self, storage: CognitionStorage):
        """
        Initialize identity manager.

        Args:
            storage: CognitionStorage for persistence
        """
        self.storage = storage

    async def get_identity(self) -> IdentityState:
        """
        Get the current identity state.

        Returns:
            Current identity state
        """
        identity = await self.storage.get_identity_state()

        # Initialize with defaults if empty
        if not identity.learning_goals:
            identity.learning_goals = DEFAULT_LEARNING_GOALS.copy()
            await self.storage.update_identity_state(identity)

        return identity

    async def record_interaction_outcome(
        self,
        outcome: str,
        topics: list[str] | None = None,
        error_type: str | None = None,
    ) -> None:
        """
        Record the outcome of an interaction.

        Updates:
        - Interaction counts
        - Performance history
        - Knowledge gaps (if error)

        Args:
            outcome: Interaction outcome (success, partial, failure)
            topics: Topics discussed
            error_type: Type of error if failed
        """
        identity = await self.storage.get_identity_state()

        # Update counts
        identity.total_interactions += 1
        if outcome == "success":
            identity.successful_interactions += 1

        # Record metric
        metric = PerformanceMetric(
            metric_name=f"interaction_{outcome}",
            value=1.0 if outcome == "success" else 0.0,
        )
        identity.performance_history.append(metric)

        # Keep only last 100 metrics
        identity.performance_history = identity.performance_history[-100:]

        # Track knowledge gaps from failures
        if outcome == "failure" and error_type:
            gap = f"Error with {error_type}"
            if gap not in identity.knowledge_gaps:
                identity.knowledge_gaps.append(gap)
                # Keep only top 10 gaps
                identity.knowledge_gaps = identity.knowledge_gaps[-10:]

        # Update in storage
        await self.storage.update_identity_state(identity)

        logger.debug(
            "Recorded interaction outcome",
            outcome=outcome,
            success_rate=identity.success_rate,
        )

    async def add_learning_goal(self, goal: str) -> None:
        """
        Add a new learning goal.

        Args:
            goal: Learning goal description
        """
        identity = await self.storage.get_identity_state()

        if goal not in identity.learning_goals:
            identity.learning_goals.append(goal)
            # Keep only top 10 goals
            identity.learning_goals = identity.learning_goals[-10:]
            await self.storage.update_identity_state(identity)
            logger.info(f"Added learning goal: {goal}")

    async def remove_learning_goal(self, goal: str) -> None:
        """
        Remove a learning goal (achieved or no longer relevant).

        Args:
            goal: Learning goal to remove
        """
        identity = await self.storage.get_identity_state()

        if goal in identity.learning_goals:
            identity.learning_goals.remove(goal)
            await self.storage.update_identity_state(identity)
            logger.info(f"Removed learning goal: {goal}")

    async def identify_knowledge_gap(self, gap: str) -> None:
        """
        Identify a knowledge gap from failed interaction.

        Args:
            gap: Description of the knowledge gap
        """
        identity = await self.storage.get_identity_state()

        if gap not in identity.knowledge_gaps:
            identity.knowledge_gaps.append(gap)
            # Keep only top 10 gaps
            identity.knowledge_gaps = identity.knowledge_gaps[-10:]
            await self.storage.update_identity_state(identity)
            logger.info(f"Identified knowledge gap: {gap}")

    async def resolve_knowledge_gap(self, gap: str) -> None:
        """
        Mark a knowledge gap as resolved.

        Args:
            gap: Description of the resolved gap
        """
        identity = await self.storage.get_identity_state()

        if gap in identity.knowledge_gaps:
            identity.knowledge_gaps.remove(gap)
            await self.storage.update_identity_state(identity)
            logger.info(f"Resolved knowledge gap: {gap}")

    async def update_value(self, value_name: str, score: float) -> None:
        """
        Update a core value score based on feedback.

        Args:
            value_name: Name of the value (accuracy, helpfulness, clarity)
            score: New score (0-1)
        """
        identity = await self.storage.get_identity_state()

        # Blend with existing value (weighted average)
        current = identity.values.get(value_name, 0.5)
        blended = current * 0.9 + score * 0.1  # Slow adaptation
        identity.values[value_name] = max(0.0, min(1.0, blended))

        await self.storage.update_identity_state(identity)
        logger.debug(f"Updated value {value_name}: {current:.2f} -> {blended:.2f}")

    async def get_performance_summary(self) -> dict[str, Any]:
        """
        Get a summary of recent performance.

        Returns:
            Dict with performance metrics
        """
        identity = await self.storage.get_identity_state()

        # Calculate recent success rate (last 20 interactions)
        recent = identity.performance_history[-20:]
        recent_successes = sum(1 for m in recent if m.value == 1.0)
        recent_rate = recent_successes / len(recent) if recent else 0.0

        return {
            "total_interactions": identity.total_interactions,
            "overall_success_rate": identity.success_rate,
            "recent_success_rate": recent_rate,
            "learning_goals": identity.learning_goals,
            "knowledge_gaps": identity.knowledge_gaps,
            "values": identity.values,
        }

    def analyze_outcome(
        self,
        messages: list[dict[str, Any]],
        reflection_result: str | None,
    ) -> str:
        """
        Analyze messages to determine interaction outcome.

        Args:
            messages: Conversation messages
            reflection_result: Optional reflection assessment

        Returns:
            Outcome string: "success", "partial", or "failure"
        """
        # Use reflection result if available
        if reflection_result:
            if reflection_result == "satisfied":
                return "success"
            elif reflection_result == "need_more":
                return "partial"
            elif reflection_result == "blocked":
                return "failure"

        # Fallback: analyze messages
        assistant_messages = [
            m for m in messages
            if m.get("role") == "assistant" and m.get("content")
        ]

        if not assistant_messages:
            return "failure"

        last_response = assistant_messages[-1].get("content", "").lower()

        # Check for failure indicators
        failure_terms = ["i cannot", "i'm unable", "i don't have access", "error"]
        if any(term in last_response for term in failure_terms):
            return "failure"

        # Check for partial indicators
        partial_terms = ["however", "but i can't", "unfortunately", "limited"]
        if any(term in last_response for term in partial_terms):
            return "partial"

        return "success"
