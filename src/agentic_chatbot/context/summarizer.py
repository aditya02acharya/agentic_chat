"""Inline summarizer for context optimization.

Generates concise summaries of tool outputs for supervisor decision-making.
Uses haiku model for speed - summaries are generated inline after tool execution.

Design:
- Fast: Uses haiku, short prompts, structured output
- Focused: Summaries tailored to the task context
- Lightweight: Minimal tokens, just key findings
"""

from __future__ import annotations

import json
from typing import Any

from agentic_chatbot.context.models import DataChunk, DataSummary
from agentic_chatbot.utils.llm import LLMClient
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


# Prompt for fast summarization
SUMMARIZER_SYSTEM = """You are a fast summarizer. Extract key findings from tool output.

Output JSON only:
{
  "executive_summary": "One sentence summary",
  "key_findings": ["Finding 1", "Finding 2", "Finding 3"]
}

Rules:
- Maximum 3 key findings
- Each finding under 20 words
- Executive summary under 15 words
- Focus on facts relevant to the task
- If no useful data, set key_findings to empty list"""

SUMMARIZER_PROMPT = """Task: {task_description}

Tool: {source_type}
Output:
{content}

Summarize the key findings relevant to the task."""


class InlineSummarizer:
    """
    Fast inline summarizer for tool outputs.

    Generates DataSummary from DataChunk immediately after tool execution.
    Uses haiku for speed - adds minimal latency to the flow.
    """

    def __init__(self, llm_client: LLMClient | None = None):
        """
        Initialize summarizer.

        Args:
            llm_client: LLM client (creates default if not provided)
        """
        self._client = llm_client or LLMClient()

    async def summarize(
        self,
        chunk: DataChunk,
        task_description: str = "",
    ) -> DataSummary:
        """
        Generate summary from a data chunk.

        Args:
            chunk: Raw data chunk to summarize
            task_description: Context about what task this is for

        Returns:
            DataSummary with key findings
        """
        # Handle empty or error content
        if not chunk.content or chunk.content.strip() == "":
            return DataSummary(
                source_id=chunk.source_id,
                source_type=chunk.source_type,
                key_findings=[],
                executive_summary="No data returned",
                has_results=False,
                task_description=task_description,
            )

        # Truncate very long content for summarization
        content = chunk.content
        if len(content) > 3000:
            content = content[:3000] + "\n... [truncated for summarization]"

        prompt = SUMMARIZER_PROMPT.format(
            task_description=task_description or "General information retrieval",
            source_type=chunk.source_type,
            content=content,
        )

        try:
            response = await self._client.complete(
                prompt=prompt,
                system=SUMMARIZER_SYSTEM,
                model="haiku",  # Fast model for inline summarization
                max_tokens=200,  # Keep it short
                temperature=0.0,
            )

            # Parse JSON response
            result = self._parse_summary_response(response.content)

            logger.debug(
                "Inline summary generated",
                source_id=chunk.source_id,
                findings_count=len(result.get("key_findings", [])),
            )

            return DataSummary(
                source_id=chunk.source_id,
                source_type=chunk.source_type,
                key_findings=result.get("key_findings", []),
                executive_summary=result.get("executive_summary", "Summary generated"),
                has_results=True,
                task_description=task_description,
            )

        except Exception as e:
            logger.warning(f"Summarization failed, using fallback: {e}")
            return self._fallback_summary(chunk, task_description)

    def _parse_summary_response(self, response: str) -> dict[str, Any]:
        """Parse JSON response from LLM."""
        # Try to extract JSON
        response = response.strip()

        # Handle markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()

        # Find JSON object
        if "{" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            if end > start:
                response = response[start:end]

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"key_findings": [], "executive_summary": "Unable to parse summary"}

    def _fallback_summary(
        self,
        chunk: DataChunk,
        task_description: str,
    ) -> DataSummary:
        """Create fallback summary without LLM."""
        # Simple extractive summary - first 100 chars
        content = chunk.content.strip()
        if len(content) > 100:
            executive = content[:100] + "..."
        else:
            executive = content

        return DataSummary(
            source_id=chunk.source_id,
            source_type=chunk.source_type,
            key_findings=[f"Data retrieved from {chunk.source_type}"],
            executive_summary=executive,
            has_results=bool(content),
            task_description=task_description,
        )


async def summarize_tool_output(
    chunk: DataChunk,
    task_description: str = "",
    client: LLMClient | None = None,
) -> DataSummary:
    """
    Convenience function for inline summarization.

    Args:
        chunk: Data chunk to summarize
        task_description: Task context
        client: Optional LLM client

    Returns:
        DataSummary
    """
    summarizer = InlineSummarizer(client)
    return await summarizer.summarize(chunk, task_description)
