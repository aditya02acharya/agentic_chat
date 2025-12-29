"""Theory of Mind: User modeling and preference learning.

This component tracks user characteristics and preferences to
personalize responses and anticipate user needs.

Key features:
- Expertise level inference (novice, intermediate, expert)
- Communication style detection (concise, detailed, technical)
- Domain interest tracking
- Preference learning (prefers examples, prefers code, etc.)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from agentic_chatbot.cognition.models import (
    UserProfile,
    ExpertiseLevel,
    CommunicationStyle,
)
from agentic_chatbot.utils.logging import get_logger

if TYPE_CHECKING:
    from agentic_chatbot.cognition.storage import CognitionStorage


logger = get_logger(__name__)


# Keywords for domain detection
DOMAIN_KEYWORDS = {
    "programming": ["code", "function", "class", "api", "debug", "error", "syntax", "programming"],
    "data_science": ["data", "model", "training", "dataset", "ml", "machine learning", "pandas", "numpy"],
    "web_development": ["html", "css", "javascript", "react", "frontend", "backend", "api", "rest"],
    "devops": ["docker", "kubernetes", "ci/cd", "deploy", "container", "infrastructure"],
    "databases": ["sql", "database", "query", "table", "schema", "postgresql", "mongodb"],
    "security": ["security", "authentication", "authorization", "encryption", "vulnerability"],
}

# Patterns for expertise detection
EXPERT_PATTERNS = [
    r"\boptimize\b", r"\bperformance\b", r"\bscalability\b", r"\barchitecture\b",
    r"\bdesign pattern\b", r"\brefactor\b", r"\bbest practice\b", r"\btrade-?off\b",
]
NOVICE_PATTERNS = [
    r"\bhow do I\b", r"\bwhat is\b", r"\bI'm new\b", r"\bbeginner\b",
    r"\bI don't understand\b", r"\bconfused\b", r"\bexplain\b",
]


class TheoryOfMind:
    """
    User modeling component for Theory of Mind.

    Infers user characteristics from interaction patterns
    and updates profiles for personalization.
    """

    def __init__(self, storage: CognitionStorage):
        """
        Initialize Theory of Mind.

        Args:
            storage: CognitionStorage for persistence
        """
        self.storage = storage

    async def get_user_profile(self, user_id: str) -> UserProfile:
        """
        Get or create user profile.

        Args:
            user_id: User identifier

        Returns:
            User profile (existing or newly created)
        """
        return await self.storage.get_or_create_user_profile(user_id)

    async def update_from_interaction(
        self,
        user_id: str,
        messages: list[dict[str, Any]],
        outcome: str,
    ) -> None:
        """
        Update user profile based on interaction.

        Analyzes messages to infer:
        - Expertise level changes
        - Communication style preferences
        - Domain interests
        - Behavioral preferences

        Args:
            user_id: User identifier
            messages: List of message dicts with 'role' and 'content'
            outcome: Interaction outcome (success, partial, failure)
        """
        profile = await self.storage.get_or_create_user_profile(user_id)

        # Extract user messages
        user_messages = [
            m["content"] for m in messages
            if m.get("role") == "user" and m.get("content")
        ]
        if not user_messages:
            return

        user_text = " ".join(user_messages).lower()

        # Update expertise level
        new_expertise = self._infer_expertise(user_text, profile.expertise_level)
        if new_expertise != profile.expertise_level:
            profile.expertise_level = new_expertise
            logger.debug(f"Updated expertise for {user_id}: {new_expertise.value}")

        # Update communication style
        new_style = self._infer_communication_style(user_text, messages)
        if new_style != profile.communication_style:
            profile.communication_style = new_style
            logger.debug(f"Updated communication style for {user_id}: {new_style.value}")

        # Update domain interests
        detected_domains = self._detect_domains(user_text)
        for domain in detected_domains:
            if domain not in profile.domain_interests:
                profile.domain_interests.append(domain)
        # Keep only top 10 interests
        profile.domain_interests = profile.domain_interests[-10:]

        # Update preferences
        self._update_preferences(profile, user_text, messages)

        # Update interaction count
        profile.interaction_count += 1
        if outcome == "success":
            profile.successful_interactions += 1

        # Save updated profile
        await self.storage.update_user_profile(profile)

        logger.info(
            f"Updated user profile for {user_id}",
            expertise=profile.expertise_level.value,
            style=profile.communication_style.value,
            domains=len(profile.domain_interests),
        )

    def _infer_expertise(
        self,
        text: str,
        current_level: ExpertiseLevel,
    ) -> ExpertiseLevel:
        """
        Infer expertise level from text patterns.

        Uses pattern matching to detect expert or novice signals,
        with gradual transitions to avoid oscillation.
        """
        expert_score = sum(1 for p in EXPERT_PATTERNS if re.search(p, text, re.I))
        novice_score = sum(1 for p in NOVICE_PATTERNS if re.search(p, text, re.I))

        # Strong expert signals
        if expert_score >= 2 and novice_score == 0:
            if current_level == ExpertiseLevel.INTERMEDIATE:
                return ExpertiseLevel.EXPERT
            elif current_level == ExpertiseLevel.NOVICE:
                return ExpertiseLevel.INTERMEDIATE

        # Strong novice signals
        if novice_score >= 2 and expert_score == 0:
            if current_level == ExpertiseLevel.EXPERT:
                return ExpertiseLevel.INTERMEDIATE
            elif current_level == ExpertiseLevel.INTERMEDIATE:
                return ExpertiseLevel.NOVICE

        # No strong signal, keep current
        return current_level

    def _infer_communication_style(
        self,
        text: str,
        messages: list[dict[str, Any]],
    ) -> CommunicationStyle:
        """
        Infer preferred communication style from messages.

        Analyzes:
        - Message length (short = concise, long = detailed)
        - Technical vocabulary usage
        - Explicit preferences ("briefly", "in detail", etc.)
        """
        # Check for explicit preferences
        if re.search(r"\bbrief(ly)?\b|\bshort\b|\bquick\b|\btl;?dr\b", text, re.I):
            return CommunicationStyle.CONCISE

        if re.search(r"\bdetail(ed)?\b|\bexplain\b|\bstep by step\b", text, re.I):
            return CommunicationStyle.DETAILED

        # Check technical vocabulary
        technical_terms = sum(1 for word in [
            "api", "function", "method", "class", "interface", "async",
            "thread", "process", "memory", "heap", "stack", "pointer",
        ] if word in text)

        if technical_terms >= 3:
            return CommunicationStyle.TECHNICAL

        # Analyze average message length
        user_messages = [
            m["content"] for m in messages
            if m.get("role") == "user" and m.get("content")
        ]
        if user_messages:
            avg_length = sum(len(m) for m in user_messages) / len(user_messages)
            if avg_length < 50:
                return CommunicationStyle.CONCISE
            elif avg_length > 200:
                return CommunicationStyle.DETAILED

        return CommunicationStyle.DETAILED  # Default

    def _detect_domains(self, text: str) -> list[str]:
        """
        Detect domain interests from text.

        Returns:
            List of detected domain names
        """
        detected = []
        for domain, keywords in DOMAIN_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text)
            if matches >= 2:  # At least 2 keyword matches
                detected.append(domain)
        return detected

    def _update_preferences(
        self,
        profile: UserProfile,
        text: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """
        Update behavioral preferences based on interaction patterns.
        """
        # Check if user asks for examples
        if re.search(r"\bexample\b|\bshow me\b|\bdemonstrate\b", text, re.I):
            profile.preferences["prefers_examples"] = True

        # Check if user prefers code
        if re.search(r"\bcode\b|\bsnippet\b|\bimplementat", text, re.I):
            profile.preferences["prefers_code"] = True

        # Check if user wants step-by-step
        if re.search(r"\bstep by step\b|\bstep-by-step\b|\bwalkthrough\b", text, re.I):
            profile.preferences["prefers_steps"] = True

        # Check response length preferences from assistant messages
        assistant_messages = [
            m for m in messages
            if m.get("role") == "assistant" and m.get("content")
        ]
        if assistant_messages:
            # If user continues asking after short responses, they may want more detail
            # If user thanks or moves on after long responses, they may prefer detail
            pass  # Future enhancement

    def get_personalization_hints(self, profile: UserProfile) -> dict[str, Any]:
        """
        Get personalization hints for response generation.

        Returns:
            Dict with hints for the writer/synthesizer
        """
        hints = {
            "expertise_level": profile.expertise_level.value,
            "communication_style": profile.communication_style.value,
        }

        if profile.preferences.get("prefers_examples"):
            hints["include_examples"] = True

        if profile.preferences.get("prefers_code"):
            hints["include_code"] = True

        if profile.preferences.get("prefers_steps"):
            hints["use_steps"] = True

        if profile.domain_interests:
            hints["known_domains"] = profile.domain_interests[:5]

        return hints
