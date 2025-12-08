"""Main chat flow - top-level graph."""

from typing import Any

from ..core.request_context import RequestContext
from ..nodes.context.init_node import InitializeNode
from ..nodes.context.fetch_tools_node import FetchToolsNode
from ..nodes.orchestration.supervisor_node import SupervisorNode
from ..nodes.orchestration.observe_node import ObserveNode
from ..nodes.orchestration.reflect_node import ReflectNode
from ..nodes.orchestration.blocked_node import HandleBlockedNode
from ..nodes.execution.tool_node import ExecuteToolNode
from ..nodes.execution.synthesize_node import SynthesizeNode
from ..nodes.output.write_node import WriteNode
from ..nodes.output.stream_node import StreamNode
from ..nodes.output.clarify_node import ClarifyNode
from ..nodes.workflow.parse_node import ParseWorkflowNode
from ..nodes.workflow.schedule_node import ScheduleStepsNode
from ..nodes.workflow.step_node import ExecuteStepNode
from ..nodes.workflow.collect_all_node import CollectAllResultsNode
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MainChatFlow:
    """
    Main chat flow implementing the ReACT supervisor pattern.

    Design Pattern: Composite Pattern

    The flow is a graph of nodes with transitions based on actions.
    """

    def __init__(self, ctx: RequestContext):
        self.ctx = ctx
        self._nodes = {
            "initialize": InitializeNode(ctx),
            "fetch_tools": FetchToolsNode(ctx),
            "supervisor": SupervisorNode(ctx),
            "execute_tool": ExecuteToolNode(ctx),
            "observe": ObserveNode(ctx),
            "reflect": ReflectNode(ctx),
            "synthesize": SynthesizeNode(ctx),
            "write": WriteNode(ctx),
            "stream": StreamNode(ctx),
            "clarify": ClarifyNode(ctx),
            "handle_blocked": HandleBlockedNode(ctx),
            "parse_workflow": ParseWorkflowNode(ctx),
            "schedule": ScheduleStepsNode(ctx),
            "execute_step": ExecuteStepNode(ctx),
            "collect_all": CollectAllResultsNode(ctx),
        }

        self._transitions = {
            "initialize": {"fetch_tools": "fetch_tools"},
            "fetch_tools": {"supervisor": "supervisor"},
            "supervisor": {
                "answer": "write",
                "call_tool": "execute_tool",
                "create_workflow": "parse_workflow",
                "clarify": "clarify",
            },
            "execute_tool": {"observe": "observe"},
            "observe": {"reflect": "reflect"},
            "reflect": {
                "satisfied": "synthesize",
                "need_more": "supervisor",
                "blocked": "handle_blocked",
            },
            "synthesize": {"write": "write"},
            "write": {"stream": "stream"},
            "stream": {"done": None},
            "clarify": {"stream": "stream"},
            "handle_blocked": {"write": "write"},
            "parse_workflow": {"schedule": "schedule"},
            "schedule": {"execute_step": "execute_step"},
            "execute_step": {
                "execute_step": "execute_step",
                "collect_all": "collect_all",
            },
            "collect_all": {"observe": "observe"},
        }

    async def run(self, shared: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute the main flow."""
        shared = shared or {}
        current_node = "initialize"

        while current_node:
            node = self._nodes.get(current_node)
            if not node:
                logger.error(f"Unknown node: {current_node}")
                break

            logger.debug(f"Executing node: {current_node}")
            action = await node.run(shared)

            if action == "done" or action == "error":
                break

            transitions = self._transitions.get(current_node, {})
            next_node = transitions.get(action)

            if next_node is None and action not in transitions:
                logger.warning(f"No transition for action '{action}' from '{current_node}'")
                break

            current_node = next_node

        return shared


async def execute_chat_flow(ctx: RequestContext, shared: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute the main chat flow."""
    flow = MainChatFlow(ctx)
    return await flow.run(shared)
