"""Engagement Decision: Intrinsic motivation for agent coordination.

This module determines WHEN the supervisor should engage other agents (tools/operators)
versus answering directly from its own knowledge.

The key insight: Tools and operators are other agents that require coordination overhead.
For simple queries, direct answers are more efficient and often better.

Intrinsic Motivation Factors:
1. Query complexity - Does this require external data or just reasoning?
2. Confidence in direct answer - How sure am I that I can answer well?
3. User expectations - Does the user expect research or quick answers?
4. Value alignment - Accuracy vs helpfulness trade-off

This prevents over-eager tool exploration for simple queries while ensuring
complex queries get proper agent coordination.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any
import re

if TYPE_CHECKING:
    from agentic_chatbot.cognition.models import UserProfile, IdentityState


class QueryComplexity(str, Enum):
    """Classification of query complexity for engagement decisions."""

    # Direct answer - no agent coordination needed
    TRIVIAL = "trivial"           # "What is 2+2?", "What does API stand for?"
    SIMPLE = "simple"             # "How do I create a list in Python?"

    # May need agent coordination
    MODERATE = "moderate"         # "Compare React and Vue for my project"

    # Definitely needs agent coordination
    COMPLEX = "complex"           # "Search for the latest Python 3.12 features"
    RESEARCH = "research"         # "Find current best practices for X"


class EngagementLevel(str, Enum):
    """How much agent coordination is recommended."""

    DIRECT_ANSWER = "direct_answer"       # Answer immediately, no tools
    OPTIONAL_TOOLS = "optional_tools"     # Can use tools if helpful, not required
    RECOMMENDED_TOOLS = "recommended"     # Should use tools for better answer
    REQUIRED_TOOLS = "required"           # Must use tools, can't answer without


@dataclass
class EngagementDecision:
    """Decision about whether to engage other agents (tools)."""

    complexity: QueryComplexity
    engagement_level: EngagementLevel
    confidence_in_direct_answer: float  # 0-1
    reasoning: str
    suggested_approach: str

    # For tool engagement
    suggested_categories: list[str]     # Which tool categories might help
    skip_discovery: bool                # Can skip browse_tools if we know what's needed


# Patterns that suggest the LLM can answer directly
DIRECT_ANSWER_PATTERNS = [
    # Definitional questions
    r"\bwhat\s+(?:is|are|does)\s+(?:a|an|the)?\s*\w+\b",
    r"\bdefine\s+\w+",
    r"\bexplain\s+(?:the\s+)?(?:concept|idea|difference)",

    # General knowledge
    r"\bhow\s+(?:do|does|can)\s+(?:I|you|we)\s+\w+",
    r"\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:best|common|typical)\s+(?:way|practice)",

    # Opinions/reasoning
    r"\bwhy\s+(?:is|are|do|does|should)",
    r"\bshould\s+I\s+\w+",
    r"\bwhat\s+do\s+you\s+(?:think|recommend)",

    # Simple coding
    r"\bhow\s+(?:to|do\s+I)\s+(?:write|create|make)\s+(?:a|an)\s+\w+",
    r"\bwrite\s+(?:a|an)\s+(?:function|code|script)\s+(?:that|to|for)",

    # Comparisons (if general)
    r"\bcompare\s+\w+\s+(?:and|vs|versus)\s+\w+$",
]

# Patterns that REQUIRE external data (tools needed)
EXTERNAL_DATA_PATTERNS = [
    # Explicit search/research
    r"\bsearch\s+(?:for|the)",
    r"\bfind\s+(?:me|the|information)",
    r"\blook\s+up\b",
    r"\bresearch\b",

    # Current/real-time data
    r"\b(?:latest|newest|recent|current|today|now)\b",
    r"\b20(?:2[4-9]|[3-9]\d)\b",  # Years 2024+
    r"\bthis\s+(?:week|month|year)\b",

    # Specific external resources
    r"\bfrom\s+(?:the\s+)?(?:web|internet|documentation|docs)",
    r"\bcheck\s+(?:the|if)\b",
    r"\bverify\b",

    # Uploaded documents
    r"\b(?:the|my|this)\s+(?:document|file|pdf|upload)",
    r"\bin\s+the\s+(?:attached|uploaded)",
]

# Signals that user wants quick/direct answers
QUICK_ANSWER_SIGNALS = [
    r"\bquick(?:ly)?\b",
    r"\bbrief(?:ly)?\b",
    r"\bin\s+short\b",
    r"\bjust\s+tell\s+me\b",
    r"\bsimple\s+(?:answer|explanation)\b",
    r"\bTL;?DR\b",
]


class EngagementDecider:
    """
    Decides whether to engage other agents (tools) based on intrinsic motivation.

    Uses query analysis, user context, and system values to determine
    the optimal engagement strategy.
    """

    def __init__(
        self,
        user_profile: UserProfile | None = None,
        identity: IdentityState | None = None,
    ):
        self.user_profile = user_profile
        self.identity = identity

    def decide(
        self,
        query: str,
        has_documents: bool = False,
        tool_results_so_far: list[Any] | None = None,
    ) -> EngagementDecision:
        """
        Decide whether to engage other agents for this query.

        Args:
            query: The user's query
            has_documents: Whether user has uploaded documents
            tool_results_so_far: Results from previous tool calls (if any)

        Returns:
            EngagementDecision with complexity, level, and reasoning
        """
        query_lower = query.lower().strip()

        # Already have tool results? Evaluate if we need more
        if tool_results_so_far:
            return self._decide_with_results(query, tool_results_so_far)

        # Check for explicit external data needs
        needs_external = self._needs_external_data(query_lower, has_documents)
        if needs_external:
            return EngagementDecision(
                complexity=QueryComplexity.RESEARCH if "search" in query_lower else QueryComplexity.COMPLEX,
                engagement_level=EngagementLevel.REQUIRED_TOOLS,
                confidence_in_direct_answer=0.1,
                reasoning=needs_external,
                suggested_approach="Use tool discovery to find appropriate data sources",
                suggested_categories=self._suggest_categories(query_lower),
                skip_discovery=False,
            )

        # Check for quick answer signals
        wants_quick = self._wants_quick_answer(query_lower)

        # Check if this is a direct-answerable query
        direct_confidence = self._assess_direct_answer_confidence(query_lower)

        # User expertise affects expectations
        expertise_factor = self._get_expertise_factor()

        # System values affect trade-off
        value_factor = self._get_value_factor()

        # Combine factors
        adjusted_confidence = direct_confidence * expertise_factor * value_factor
        if wants_quick:
            adjusted_confidence *= 1.2  # Boost confidence for quick answers

        # Clamp to [0, 1]
        adjusted_confidence = min(1.0, max(0.0, adjusted_confidence))

        # Determine engagement level
        if adjusted_confidence >= 0.8:
            return EngagementDecision(
                complexity=QueryComplexity.TRIVIAL if adjusted_confidence >= 0.95 else QueryComplexity.SIMPLE,
                engagement_level=EngagementLevel.DIRECT_ANSWER,
                confidence_in_direct_answer=adjusted_confidence,
                reasoning="High confidence in direct answer based on query type and context",
                suggested_approach="Answer directly using your knowledge. No tool coordination needed.",
                suggested_categories=[],
                skip_discovery=True,
            )
        elif adjusted_confidence >= 0.5:
            return EngagementDecision(
                complexity=QueryComplexity.MODERATE,
                engagement_level=EngagementLevel.OPTIONAL_TOOLS,
                confidence_in_direct_answer=adjusted_confidence,
                reasoning="Moderate confidence - can answer directly but tools may improve quality",
                suggested_approach="Consider answering directly first. Use tools only if you feel uncertain.",
                suggested_categories=self._suggest_categories(query_lower),
                skip_discovery=True,  # Don't waste time on discovery for optional
            )
        else:
            return EngagementDecision(
                complexity=QueryComplexity.COMPLEX,
                engagement_level=EngagementLevel.RECOMMENDED_TOOLS,
                confidence_in_direct_answer=adjusted_confidence,
                reasoning="Low confidence in direct answer - tool coordination recommended",
                suggested_approach="Use tools to gather information before answering",
                suggested_categories=self._suggest_categories(query_lower),
                skip_discovery=False,
            )

    def _needs_external_data(self, query: str, has_documents: bool) -> str | None:
        """Check if query explicitly needs external data."""
        for pattern in EXTERNAL_DATA_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return f"Query matches external data pattern: {pattern}"

        # Document queries need document tools
        if has_documents and any(kw in query for kw in ["document", "file", "uploaded", "pdf"]):
            return "Query references uploaded documents"

        return None

    def _wants_quick_answer(self, query: str) -> bool:
        """Check if user wants a quick/direct answer."""
        return any(re.search(p, query, re.IGNORECASE) for p in QUICK_ANSWER_SIGNALS)

    def _assess_direct_answer_confidence(self, query: str) -> float:
        """Assess confidence that LLM can answer directly without tools."""
        confidence = 0.5  # Base confidence

        # Check for direct-answerable patterns
        direct_matches = sum(
            1 for p in DIRECT_ANSWER_PATTERNS
            if re.search(p, query, re.IGNORECASE)
        )
        if direct_matches > 0:
            confidence += min(0.4, direct_matches * 0.15)

        # Short queries are often simpler
        word_count = len(query.split())
        if word_count <= 10:
            confidence += 0.1
        elif word_count >= 30:
            confidence -= 0.1

        # Questions vs commands
        if query.rstrip().endswith("?"):
            confidence += 0.05  # Questions are often direct-answerable

        # Penalize specific/concrete requirements
        if any(kw in query for kw in ["specific", "exact", "precise", "actual"]):
            confidence -= 0.2

        return max(0.0, min(1.0, confidence))

    def _get_expertise_factor(self) -> float:
        """Get factor based on user expertise."""
        if not self.user_profile:
            return 1.0

        # Experts may want more depth (lower direct confidence)
        # Novices may prefer simpler direct answers
        expertise = self.user_profile.expertise_level.value
        if expertise == "expert":
            return 0.9  # Slightly lower - experts may want more research
        elif expertise == "novice":
            return 1.1  # Slightly higher - novices prefer simpler answers
        return 1.0

    def _get_value_factor(self) -> float:
        """Get factor based on system values."""
        if not self.identity:
            return 1.0

        values = self.identity.values
        accuracy = values.get("accuracy", 0.9)
        helpfulness = values.get("helpfulness", 0.95)

        # If helpfulness > accuracy, prefer direct answers
        # If accuracy > helpfulness, prefer more thorough research
        if helpfulness > accuracy:
            return 1.05  # Slightly boost direct answer confidence
        elif accuracy > helpfulness:
            return 0.95  # Slightly reduce (favor research)
        return 1.0

    def _suggest_categories(self, query: str) -> list[str]:
        """Suggest tool categories based on query content."""
        categories = []

        if any(kw in query for kw in ["search", "find", "look", "web", "internet"]):
            categories.append("information_retrieval")
        if any(kw in query for kw in ["document", "file", "pdf", "uploaded"]):
            categories.append("document_management")
        if any(kw in query for kw in ["code", "run", "execute", "script"]):
            categories.append("code_execution")
        if any(kw in query for kw in ["analyze", "data", "chart", "graph"]):
            categories.append("data_analysis")

        return categories or ["uncategorized"]

    def _decide_with_results(
        self,
        query: str,
        tool_results: list[Any],
    ) -> EngagementDecision:
        """Decide next step when we already have some tool results."""
        # If we have results, evaluate if they're sufficient
        result_count = len(tool_results)

        if result_count >= 3:
            # We've done significant research, probably time to answer
            return EngagementDecision(
                complexity=QueryComplexity.COMPLEX,
                engagement_level=EngagementLevel.DIRECT_ANSWER,
                confidence_in_direct_answer=0.8,
                reasoning=f"Have {result_count} tool results - sufficient for answering",
                suggested_approach="Synthesize the collected information and answer",
                suggested_categories=[],
                skip_discovery=True,
            )
        else:
            # May need more, but don't force it
            return EngagementDecision(
                complexity=QueryComplexity.MODERATE,
                engagement_level=EngagementLevel.OPTIONAL_TOOLS,
                confidence_in_direct_answer=0.6,
                reasoning=f"Have {result_count} result(s) - evaluate if more needed",
                suggested_approach="Assess if current results are sufficient, or gather more",
                suggested_categories=[],
                skip_discovery=True,
            )


def create_engagement_context(
    query: str,
    user_profile: "UserProfile | None" = None,
    identity: "IdentityState | None" = None,
    has_documents: bool = False,
    tool_results: list[Any] | None = None,
) -> str:
    """
    Create engagement context text for the supervisor prompt.

    This provides intrinsic motivation guidance about whether
    to engage other agents (tools) or answer directly.
    """
    decider = EngagementDecider(user_profile=user_profile, identity=identity)
    decision = decider.decide(query, has_documents=has_documents, tool_results_so_far=tool_results)

    lines = [
        "## Agent Coordination Guidance",
        "",
        f"**Query Complexity**: {decision.complexity.value}",
        f"**Recommended Engagement**: {decision.engagement_level.value}",
        f"**Direct Answer Confidence**: {decision.confidence_in_direct_answer:.0%}",
        "",
        f"**Reasoning**: {decision.reasoning}",
        "",
        f"**Suggested Approach**: {decision.suggested_approach}",
    ]

    if decision.suggested_categories:
        lines.append(f"**Relevant Tool Categories**: {', '.join(decision.suggested_categories)}")

    if decision.skip_discovery:
        lines.append("")
        lines.append("*Skip tool discovery - answer directly or use known tools.*")

    return "\n".join(lines)
