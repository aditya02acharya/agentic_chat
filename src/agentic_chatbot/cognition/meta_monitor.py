"""Meta Monitor: Self-reflection and confidence monitoring.

This component provides meta-cognitive capabilities:
- Confidence scoring for responses
- Error pattern analysis
- Strategy effectiveness tracking
- Self-reflection prompts

The meta monitor observes the system's behavior and provides
insights for continuous improvement.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agentic_chatbot.utils.logging import get_logger

if TYPE_CHECKING:
    from agentic_chatbot.cognition.storage import CognitionStorage


logger = get_logger(__name__)


@dataclass
class ConfidenceAssessment:
    """Assessment of confidence in a response or decision."""

    score: float  # 0-1 confidence score
    factors: dict[str, float]  # Contributing factors
    warnings: list[str]  # Potential issues
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "score": self.score,
            "factors": self.factors,
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ErrorPattern:
    """Pattern of errors for analysis."""

    pattern_type: str
    frequency: int
    last_seen: datetime
    examples: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pattern_type": self.pattern_type,
            "frequency": self.frequency,
            "last_seen": self.last_seen.isoformat(),
            "examples": self.examples,
        }


class MetaMonitor:
    """
    Meta-cognitive monitoring component.

    Provides self-reflection capabilities and confidence assessment
    for improving system performance over time.
    """

    def __init__(self, storage: CognitionStorage):
        """
        Initialize meta monitor.

        Args:
            storage: CognitionStorage for persistence
        """
        self.storage = storage
        self._error_patterns: dict[str, ErrorPattern] = {}

    def assess_confidence(
        self,
        decision: dict[str, Any],
        context: dict[str, Any],
    ) -> ConfidenceAssessment:
        """
        Assess confidence in a supervisor decision.

        Evaluates multiple factors:
        - Information completeness
        - Query clarity
        - Tool availability
        - Previous similar interactions

        Args:
            decision: Supervisor decision dict
            context: Current context dict

        Returns:
            ConfidenceAssessment with score and factors
        """
        factors = {}
        warnings = []

        # Factor 1: Decision clarity
        reasoning = decision.get("reasoning", "")
        if len(reasoning) > 50:
            factors["reasoning_quality"] = 0.8
        elif len(reasoning) > 20:
            factors["reasoning_quality"] = 0.6
        else:
            factors["reasoning_quality"] = 0.4
            warnings.append("Reasoning is brief, may lack depth")

        # Factor 2: Action specificity
        action = decision.get("action", "")
        if action == "ANSWER" and decision.get("response"):
            factors["action_specificity"] = 0.9
        elif action == "CALL_TOOL" and decision.get("operator"):
            factors["action_specificity"] = 0.8
        elif action == "CREATE_WORKFLOW" and decision.get("goal"):
            factors["action_specificity"] = 0.7
        else:
            factors["action_specificity"] = 0.5
            warnings.append("Action details may be incomplete")

        # Factor 3: Context richness
        tool_results = context.get("tool_results", [])
        if len(tool_results) >= 2:
            factors["context_richness"] = 0.9
        elif len(tool_results) == 1:
            factors["context_richness"] = 0.7
        else:
            factors["context_richness"] = 0.5

        # Factor 4: Iteration count (lower is better)
        iteration = context.get("iteration", 0)
        if iteration <= 1:
            factors["efficiency"] = 0.9
        elif iteration <= 3:
            factors["efficiency"] = 0.7
        else:
            factors["efficiency"] = 0.5
            warnings.append("Multiple iterations may indicate difficulty")

        # Calculate overall score (weighted average)
        weights = {
            "reasoning_quality": 0.3,
            "action_specificity": 0.3,
            "context_richness": 0.2,
            "efficiency": 0.2,
        }
        score = sum(factors[k] * weights[k] for k in factors)

        return ConfidenceAssessment(
            score=score,
            factors=factors,
            warnings=warnings,
            timestamp=datetime.utcnow(),
        )

    def analyze_error(
        self,
        error_type: str,
        error_message: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Analyze an error for pattern detection.

        Args:
            error_type: Type of error
            error_message: Error message
            context: Context when error occurred

        Returns:
            Analysis dict with pattern info and suggestions
        """
        # Track error pattern
        if error_type in self._error_patterns:
            pattern = self._error_patterns[error_type]
            pattern.frequency += 1
            pattern.last_seen = datetime.utcnow()
            if len(pattern.examples) < 5:
                pattern.examples.append(error_message[:100])
        else:
            self._error_patterns[error_type] = ErrorPattern(
                pattern_type=error_type,
                frequency=1,
                last_seen=datetime.utcnow(),
                examples=[error_message[:100]],
            )

        pattern = self._error_patterns[error_type]

        # Generate suggestions based on pattern
        suggestions = []
        if pattern.frequency >= 3:
            suggestions.append(f"Recurring error type: {error_type} (seen {pattern.frequency} times)")
        if "timeout" in error_type.lower():
            suggestions.append("Consider breaking task into smaller steps")
        if "not found" in error_message.lower():
            suggestions.append("Verify resource availability before operation")
        if "permission" in error_message.lower():
            suggestions.append("Check access permissions for requested operation")

        return {
            "pattern_type": error_type,
            "frequency": pattern.frequency,
            "is_recurring": pattern.frequency >= 3,
            "suggestions": suggestions,
            "similar_errors": pattern.examples,
        }

    def generate_self_reflection_prompt(
        self,
        interaction_summary: str,
        outcome: str,
        confidence: ConfidenceAssessment,
    ) -> str:
        """
        Generate a self-reflection prompt for learning.

        Args:
            interaction_summary: Summary of the interaction
            outcome: Outcome (success, partial, failure)
            confidence: Confidence assessment

        Returns:
            Reflection prompt string
        """
        prompt_parts = [
            "## Self-Reflection on Recent Interaction\n",
            f"**Summary:** {interaction_summary}\n",
            f"**Outcome:** {outcome}\n",
            f"**Confidence Score:** {confidence.score:.2f}\n",
        ]

        if confidence.warnings:
            prompt_parts.append(f"**Warnings:** {', '.join(confidence.warnings)}\n")

        prompt_parts.append("\n**Reflection Questions:**\n")

        if outcome == "failure":
            prompt_parts.extend([
                "1. What information was missing that led to this failure?",
                "2. Were there alternative approaches that could have succeeded?",
                "3. What can be learned to prevent similar failures?",
            ])
        elif outcome == "partial":
            prompt_parts.extend([
                "1. What aspects were handled well?",
                "2. What aspects could be improved?",
                "3. How can the response be more complete next time?",
            ])
        else:
            prompt_parts.extend([
                "1. What made this interaction successful?",
                "2. Can this approach be generalized to similar queries?",
                "3. Were there any inefficiencies that could be reduced?",
            ])

        return "\n".join(prompt_parts)

    def get_strategy_recommendations(
        self,
        query: str,
        user_profile: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        Get strategy recommendations based on query and user.

        Args:
            query: Current user query
            user_profile: Optional user profile dict

        Returns:
            List of strategy recommendations
        """
        recommendations = []
        query_lower = query.lower()

        # Complexity-based recommendations
        if any(word in query_lower for word in ["compare", "vs", "difference"]):
            recommendations.append("Consider structured comparison format")

        if any(word in query_lower for word in ["step", "how to", "guide"]):
            recommendations.append("Use numbered steps for clarity")

        if any(word in query_lower for word in ["why", "explain", "reason"]):
            recommendations.append("Provide clear reasoning and examples")

        if any(word in query_lower for word in ["best", "recommend", "should"]):
            recommendations.append("Consider trade-offs and user context")

        # User-based recommendations
        if user_profile:
            expertise = user_profile.get("expertise_level", "intermediate")
            if expertise == "novice":
                recommendations.append("Use simpler language and more examples")
            elif expertise == "expert":
                recommendations.append("Be concise, focus on key details")

            style = user_profile.get("communication_style", "detailed")
            if style == "concise":
                recommendations.append("Keep response focused and brief")
            elif style == "technical":
                recommendations.append("Include technical details and code")

        return recommendations

    def get_error_patterns_summary(self) -> list[dict[str, Any]]:
        """
        Get summary of detected error patterns.

        Returns:
            List of error pattern dicts
        """
        patterns = sorted(
            self._error_patterns.values(),
            key=lambda p: p.frequency,
            reverse=True,
        )
        return [p.to_dict() for p in patterns[:10]]
