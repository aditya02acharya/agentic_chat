"""Supervisor node implementing ReACT loop."""

from typing import Any

from pydantic import BaseModel, Field

from agentic_chatbot.config.prompts import SUPERVISOR_SYSTEM_PROMPT, SUPERVISOR_DECISION_PROMPT
from agentic_chatbot.core.supervisor import SupervisorDecision, SupervisorState
from agentic_chatbot.events.models import SupervisorThinkingEvent, SupervisorDecidedEvent
from agentic_chatbot.nodes.base import AsyncBaseNode
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.utils.structured_llm import StructuredLLMCaller, StructuredOutputError
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

    async def exec_async(self, prep_res: dict[str, Any]) -> SupervisorDecision:
        """Execute supervisor decision-making."""
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
            decision = await caller.call(
                prompt=prompt,
                response_model=SupervisorDecision,
                system=system,
                model="sonnet",
            )
            return decision

        except StructuredOutputError as e:
            # Fallback to CLARIFY action
            logger.error(f"Supervisor decision failed: {e.attempts}")
            return SupervisorDecision(
                action="CLARIFY",
                reasoning="Unable to determine appropriate action due to processing error",
                question="Could you please rephrase your request?",
            )

    async def post_async(
        self,
        shared: dict[str, Any],
        prep_res: dict[str, Any],
        exec_res: SupervisorDecision,
    ) -> str:
        """Store decision and return action for routing."""
        state: SupervisorState = prep_res["state"]

        # Store decision
        shared["supervisor"]["current_decision"] = exec_res
        state.current_decision = exec_res

        # Emit decided event
        action_messages = {
            "ANSWER": "I can answer this directly",
            "CALL_TOOL": f"Using {exec_res.operator or 'tool'} to get information",
            "CREATE_WORKFLOW": "Creating a multi-step plan",
            "CLARIFY": "I need some clarification",
        }
        await self.emit_event(
            shared,
            SupervisorDecidedEvent.create(
                action=exec_res.action,
                message=action_messages.get(exec_res.action, "Processing..."),
                request_id=shared.get("request_id"),
            ),
        )

        logger.info(
            "Supervisor decided",
            action=exec_res.action,
            reasoning=exec_res.reasoning[:100],
        )

        # Map to flow action
        return exec_res.action.lower()

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
