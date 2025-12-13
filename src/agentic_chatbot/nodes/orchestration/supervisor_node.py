"""Supervisor node implementing ReACT loop."""

from typing import Any

from pydantic import BaseModel, Field

from agentic_chatbot.config.models import TokenUsage
from agentic_chatbot.config.prompts import SUPERVISOR_SYSTEM_PROMPT, SUPERVISOR_DECISION_PROMPT
from agentic_chatbot.core.supervisor import SupervisorDecision, SupervisorState
from agentic_chatbot.events.models import SupervisorThinkingEvent, SupervisorDecidedEvent
from agentic_chatbot.nodes.base import AsyncBaseNode
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.utils.structured_llm import StructuredLLMCaller, StructuredOutputError, StructuredResult
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class SupervisorNode(AsyncBaseNode):
    """
    Central decision-making agent using ReACT pattern.

    Type: Orchestration Node (with internal state machine)
    Max Iterations: 5

    The Supervisor:
    - THINKS: Analyzes the user query
    - REASONS: Determines what information/actions are needed
    - PLANS: Decides action type (direct answer, single tool, workflow, clarify)
    - ACTS: Returns the action for flow routing

    Returns: "answer" | "call_tool" | "workflow" | "clarify"
    """

    node_name = "supervisor"
    description = "ReACT supervisor for decision making"

    async def prep_async(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Prepare supervisor context."""
        # Emit thinking event
        await self.emit_event(
            shared,
            SupervisorThinkingEvent.create(
                "Analyzing your question...",
                request_id=shared.get("request_id"),
            ),
        )

        # Get supervisor state
        supervisor_state = shared.get("supervisor", {})
        if isinstance(supervisor_state, dict) and "state" in supervisor_state:
            state = supervisor_state["state"]
        else:
            state = SupervisorState()
            shared.setdefault("supervisor", {})["state"] = state

        # Build context for decision
        context = {
            "query": shared.get("user_query", ""),
            "conversation_context": self._format_conversation_context(shared),
            "tool_summaries": self._get_tool_summaries(shared),
            "actions_this_turn": state.actions_this_turn,
            "iteration": state.iteration,
            "state": state,
        }

        return context

    async def exec_async(self, prep_res: dict[str, Any]) -> StructuredResult[SupervisorDecision]:
        """Execute supervisor decision-making with extended thinking."""
        caller = StructuredLLMCaller(max_retries=3)

        # Build prompts
        tool_summaries = prep_res["tool_summaries"]
        system = SUPERVISOR_SYSTEM_PROMPT.format(tool_summaries=tool_summaries)

        prompt = SUPERVISOR_DECISION_PROMPT.format(
            query=prep_res["query"],
            conversation_context=prep_res["conversation_context"],
            actions_this_turn="\n".join(prep_res["actions_this_turn"]) or "None",
        )

        try:
            # Use extended thinking for complex reasoning
            result = await caller.call_with_usage(
                prompt=prompt,
                response_model=SupervisorDecision,
                system=system,
                model="thinking",
                enable_thinking=True,
                thinking_budget=10000,
            )
            return result

        except StructuredOutputError as e:
            # Fallback to CLARIFY action
            logger.error(f"Supervisor decision failed: {e.attempts}")
            return StructuredResult(
                data=SupervisorDecision(
                    action="CLARIFY",
                    reasoning="Unable to determine appropriate action due to processing error",
                    question="Could you please rephrase your request?",
                ),
                usage=TokenUsage(),
            )

    async def post_async(
        self,
        shared: dict[str, Any],
        prep_res: dict[str, Any],
        exec_res: StructuredResult[SupervisorDecision],
    ) -> str:
        """Store decision, track token usage, and return action for routing."""
        state: SupervisorState = prep_res["state"]

        # Extract decision and token usage from result
        decision = exec_res.data
        token_usage = exec_res.usage

        # Store decision
        shared["supervisor"]["current_decision"] = decision
        state.current_decision = decision

        # Track token usage in shared state
        current_usage = shared.get("token_usage")
        if current_usage is None:
            current_usage = TokenUsage()
        shared["token_usage"] = current_usage + token_usage

        # Log thinking content if available (for debugging)
        if exec_res.thinking_content:
            logger.debug(
                "Supervisor thinking",
                thinking_preview=exec_res.thinking_content[:200],
            )

        # Emit decided event
        action_messages = {
            "ANSWER": "I can answer this directly",
            "CALL_TOOL": f"Using {decision.operator or 'tool'} to get information",
            "CREATE_WORKFLOW": "Creating a multi-step plan",
            "CLARIFY": "I need some clarification",
        }
        await self.emit_event(
            shared,
            SupervisorDecidedEvent.create(
                action=decision.action,
                message=action_messages.get(decision.action, "Processing..."),
                request_id=shared.get("request_id"),
            ),
        )

        logger.info(
            "Supervisor decided",
            action=decision.action,
            reasoning=decision.reasoning[:100],
            thinking_tokens=token_usage.thinking_tokens,
        )

        # Map to flow action
        return decision.action.lower()

    def _format_conversation_context(self, shared: dict[str, Any]) -> str:
        """Format conversation context for prompt."""
        memory = shared.get("memory")
        if memory and hasattr(memory, "format_for_prompt"):
            return memory.format_for_prompt()

        # Fallback: format from shared directly
        recent = shared.get("recent_messages", [])
        if recent:
            return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent)

        return "No previous conversation."

    def _get_tool_summaries(self, shared: dict[str, Any]) -> str:
        """Get formatted tool summaries."""
        # Try MCP registry
        registry = shared.get("mcp", {}).get("server_registry")
        if registry and hasattr(registry, "get_tool_summaries_text"):
            return registry.get_tool_summaries_text()

        # Fallback: use operator registry
        return OperatorRegistry.get_operators_text()
