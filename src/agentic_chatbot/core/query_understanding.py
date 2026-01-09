"""Query Understanding Stage.

This module provides deep query analysis before action is taken.
It extracts goals, scope, ecology, and determines if clarification is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    """High-level intent classification."""
    INFORMATION_SEEKING = "information_seeking"  # User wants to know something
    TASK_EXECUTION = "task_execution"  # User wants something done
    EXPLORATION = "exploration"  # User is exploring/brainstorming
    CLARIFICATION = "clarification"  # User is clarifying previous exchange
    FEEDBACK = "feedback"  # User is providing feedback
    SOCIAL = "social"  # Greeting, thanks, etc.


class QueryComplexity(str, Enum):
    """Complexity assessment for resource allocation."""
    TRIVIAL = "trivial"  # Can answer directly, no tools needed
    SIMPLE = "simple"  # Single tool/operation
    MODERATE = "moderate"  # Multiple operations, some coordination
    COMPLEX = "complex"  # Multi-step workflow, significant reasoning
    AMBIGUOUS = "ambiguous"  # Cannot determine without clarification


class QueryUnderstanding(BaseModel):
    """
    Deep understanding of user query.

    This is extracted before any action is taken and guides
    the entire interaction flow.
    """

    # Core understanding
    raw_query: str = Field(description="Original user query")
    reformulated_query: str = Field(description="Clarified/expanded version of query")

    # Goals and objectives
    primary_goal: str = Field(description="Main objective the user wants to achieve")
    secondary_goals: list[str] = Field(default_factory=list, description="Supporting objectives")
    success_criteria: list[str] = Field(default_factory=list, description="How to know if goal is achieved")

    # Scope
    scope_inclusions: list[str] = Field(default_factory=list, description="What's explicitly in scope")
    scope_exclusions: list[str] = Field(default_factory=list, description="What's out of scope")
    constraints: list[str] = Field(default_factory=list, description="Limitations or requirements")

    # Query ecology (context)
    relates_to_previous: bool = Field(default=False, description="References previous conversation")
    context_dependencies: list[str] = Field(default_factory=list, description="Information needed from context")
    implicit_assumptions: list[str] = Field(default_factory=list, description="Unstated assumptions detected")

    # Classification
    intent: QueryIntent = Field(default=QueryIntent.INFORMATION_SEEKING)
    complexity: QueryComplexity = Field(default=QueryComplexity.MODERATE)

    # Confidence and clarity
    clarity_score: float = Field(default=1.0, ge=0.0, le=1.0, description="How clear the query is (0-1)")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in understanding (0-1)")

    # Clarification
    needs_clarification: bool = Field(default=False)
    clarification_questions: list[str] = Field(default_factory=list, description="Questions to ask user")
    clarification_priority: list[str] = Field(default_factory=list, description="Which questions are most important")

    # Tool hints
    suggested_tool_categories: list[str] = Field(default_factory=list, description="Categories of tools likely needed")

    def requires_tools(self) -> bool:
        """Check if this query likely requires tool usage."""
        return self.intent in (QueryIntent.INFORMATION_SEEKING, QueryIntent.TASK_EXECUTION) and \
               self.complexity not in (QueryComplexity.TRIVIAL,)

    def should_clarify_first(self) -> bool:
        """Check if clarification should be sought before proceeding."""
        return self.needs_clarification and (
            self.clarity_score < 0.5 or
            self.complexity == QueryComplexity.AMBIGUOUS or
            len(self.clarification_questions) > 0
        )


class ClarificationRequest(BaseModel):
    """Request for clarification from user."""

    question: str = Field(description="The clarification question")
    context: str = Field(default="", description="Why this clarification is needed")
    options: list[str] | None = Field(default=None, description="Suggested options if applicable")
    is_blocking: bool = Field(default=False, description="If True, cannot proceed without answer")
    source: str = Field(default="supervisor", description="Who is asking: supervisor, tool, agent")
    priority: int = Field(default=1, ge=1, le=5, description="1=highest priority")


class ClarificationResponse(BaseModel):
    """User's response to clarification."""

    question_id: str = Field(description="ID of the question being answered")
    response: str = Field(description="User's response")
    additional_context: str = Field(default="", description="Any additional context provided")


# Prompt for query understanding
QUERY_UNDERSTANDING_PROMPT = """Analyze the user's query deeply to understand their true intent.

User Query: {query}

Conversation Context:
{conversation_context}

Available Tool Categories:
{tool_categories}

Analyze and extract:

1. **Goals**: What does the user want to achieve? What would success look like?

2. **Scope**: What's included? What's explicitly or implicitly excluded? Any constraints?

3. **Ecology**: How does this relate to previous conversation? What context is assumed?

4. **Clarity**: Is the query clear enough to act on? What's ambiguous?

5. **Complexity**: How complex is this task? (trivial/simple/moderate/complex/ambiguous)

6. **Tool Hints**: What categories of tools might be needed?

If the query is vague or ambiguous, formulate conversational clarification questions
that feel natural, not interrogative. Prioritize the most important questions.

Respond with a structured analysis."""


QUERY_UNDERSTANDING_SYSTEM = """You are a query understanding specialist. Your role is to deeply
analyze user queries before any action is taken.

You excel at:
- Identifying explicit and implicit goals
- Understanding scope and constraints
- Detecting ambiguity and vagueness
- Formulating natural clarification questions
- Assessing query complexity

Be thorough but concise. When clarification is needed, ask conversational questions
that feel like natural dialogue, not an interrogation."""
