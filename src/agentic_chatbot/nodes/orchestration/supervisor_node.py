"""Supervisor node implementing ReACT loop with progressive tool discovery."""

from typing import Any

from pydantic import BaseModel, Field

from agentic_chatbot.config.prompts import SUPERVISOR_SYSTEM_PROMPT, SUPERVISOR_DECISION_PROMPT
from agentic_chatbot.core.supervisor import SupervisorDecision, SupervisorState
from agentic_chatbot.events.models import SupervisorThinkingEvent, SupervisorDecidedEvent
from agentic_chatbot.nodes.base import AsyncBaseNode
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.tools.catalog import get_catalog, ToolCategory
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

        # Build prompts with catalog overview (not full tool dump)
        tool_catalog_overview = prep_res["tool_summaries"]
        system = SUPERVISOR_SYSTEM_PROMPT.format(tool_catalog_overview=tool_catalog_overview)

        prompt = SUPERVISOR_DECISION_PROMPT.format(
            query=prep_res["query"],
            conversation_context=prep_res["conversation_context"],
            document_context=prep_res.get("document_context", "No documents uploaded."),
            tool_results=prep_res.get("tool_results", "None yet."),
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
        """Get catalog overview for progressive discovery.

        Instead of dumping all tools (which doesn't scale), we provide:
        1. A high-level overview of tool categories
        2. Instructions on how to discover tools progressively

        The supervisor uses browse_tools, search_tools, and get_tool_info
        to discover and learn about specific tools as needed.
        """
        catalog = get_catalog()

        # Populate catalog from registries if needed
        self._populate_catalog(shared, catalog)

        # Get overview (categories + counts, not individual tools)
        overview = catalog.get_overview()

        if overview.total_tools == 0:
            return "No tools currently registered. Use browse_tools to check."

        # Format as concise overview
        lines = [
            f"## Tool Catalog Overview ({overview.total_tools} tools)",
            "",
        ]

        for cat in overview.categories:
            if cat.tool_count > 0:
                groups_preview = ", ".join(cat.groups[:3])
                if len(cat.groups) > 3:
                    groups_preview += f" +{len(cat.groups) - 3} more"
                lines.append(
                    f"- **{cat.category.value}** ({cat.tool_count} tools): "
                    f"{cat.description[:60]}..."
                )
                lines.append(f"  Groups: {groups_preview}")

        lines.append("")
        lines.append("Use `browse_tools`, `search_tools`, or `get_tool_info` to discover specific tools.")

        return "\n".join(lines)

    def _populate_catalog(self, shared: dict[str, Any], catalog) -> None:
        """Populate the catalog from available registries."""
        if catalog._entries:
            return  # Already populated

        # Add operators
        for summary in OperatorRegistry.get_all_summaries():
            category = self._categorize_tool(summary["name"], summary.get("description", ""))
            catalog.register(
                name=summary["name"],
                description=summary.get("description", ""),
                category=category,
                group_id=self._get_group_id(summary["name"], category),
                is_local=False,
            )

        # Add MCP tools if available
        registry = shared.get("mcp", {}).get("server_registry")
        if registry and hasattr(registry, "get_tool_summaries_text"):
            try:
                # Note: This is sync, so we can't await. In practice,
                # the catalog would be populated async elsewhere.
                pass
            except Exception as e:
                logger.warning(f"Failed to populate MCP tools: {e}")

    def _categorize_tool(self, name: str, description: str) -> ToolCategory:
        """Determine category based on tool name and description."""
        name_lower = name.lower()
        desc_lower = description.lower()

        if any(kw in name_lower for kw in ["list", "browse", "self", "info"]):
            return ToolCategory.SYSTEM_INTROSPECTION
        if any(kw in name_lower for kw in ["document", "load_doc", "file"]):
            return ToolCategory.DOCUMENT_MANAGEMENT
        if any(kw in name_lower or kw in desc_lower for kw in ["search", "retrieve", "fetch", "rag"]):
            return ToolCategory.INFORMATION_RETRIEVAL
        if any(kw in name_lower or kw in desc_lower for kw in ["code", "execute", "run"]):
            return ToolCategory.CODE_EXECUTION
        if any(kw in name_lower or kw in desc_lower for kw in ["analyze", "data", "chart"]):
            return ToolCategory.DATA_ANALYSIS

        return ToolCategory.UNCATEGORIZED

    def _get_group_id(self, name: str, category: ToolCategory) -> str:
        """Determine group within category."""
        name_lower = name.lower()

        if category == ToolCategory.INFORMATION_RETRIEVAL:
            if "web" in name_lower:
                return "web_search"
            elif "rag" in name_lower:
                return "knowledge_base"
            return "general_search"

        if category == ToolCategory.SYSTEM_INTROSPECTION:
            if "tool" in name_lower:
                return "tool_discovery"
            return "system_info"

        return "default"
