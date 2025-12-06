"""Workflow subflow for multi-step execution."""

from pocketflow import AsyncFlow

from agentic_chatbot.nodes.workflow.parse_node import ParseWorkflowNode
from agentic_chatbot.nodes.workflow.schedule_node import ScheduleStepsNode
from agentic_chatbot.nodes.workflow.step_node import ExecuteStepNode
from agentic_chatbot.nodes.workflow.parallel_node import ExecuteParallelNode
from agentic_chatbot.nodes.workflow.collect_all_node import CollectAllResultsNode


def create_workflow_subflow() -> AsyncFlow:
    """
    Create workflow execution subflow.

    Flow:
        ParseWorkflow → ScheduleSteps → [ExecuteStep|ExecuteParallel] → CollectAll

    Note: The actual step execution loop is handled by the workflow executor
    or by repeated flow execution for simplicity.

    Returns:
        Configured AsyncFlow for workflow execution
    """
    # Create nodes
    parse_workflow = ParseWorkflowNode()
    schedule_steps = ScheduleStepsNode()
    execute_step = ExecuteStepNode()
    execute_parallel = ExecuteParallelNode()
    collect_all = CollectAllResultsNode()

    # Wire nodes
    parse_workflow >> schedule_steps

    # Schedule routes to either single step or parallel
    schedule_steps - "default" >> execute_step
    schedule_steps - "parallel" >> execute_parallel
    schedule_steps - "error" >> collect_all

    # Steps route back to schedule or to collect
    execute_step - "default" >> schedule_steps
    execute_step - "done" >> collect_all

    execute_parallel - "default" >> schedule_steps
    execute_parallel - "done" >> collect_all

    # Create flow
    flow = AsyncFlow(start=parse_workflow)

    return flow


def create_simple_workflow_subflow() -> AsyncFlow:
    """
    Create simplified workflow subflow using WorkflowExecutor.

    This version uses the WorkflowExecutor class directly
    instead of individual nodes for simpler execution.

    Flow:
        ParseWorkflow → ExecuteAll → CollectAll

    Returns:
        Configured AsyncFlow
    """
    from agentic_chatbot.nodes.base import AsyncBaseNode
    from agentic_chatbot.core.workflow_executor import WorkflowExecutor
    from agentic_chatbot.events.emitter import EventEmitter
    from typing import Any

    class ExecuteWorkflowNode(AsyncBaseNode):
        """Execute entire workflow using WorkflowExecutor."""

        node_name = "execute_workflow"
        description = "Execute complete workflow"

        async def prep_async(self, shared: dict[str, Any]) -> dict[str, Any]:
            workflow = shared.get("workflow", {})
            definition = workflow.get("definition")

            # Get event emitter
            emitter = None
            queue = shared.get("event_queue")
            if queue:
                emitter = EventEmitter(queue)

            # Get session manager
            session_manager = shared.get("mcp", {}).get("session_manager")

            return {
                "workflow": definition,
                "emitter": emitter,
                "session_manager": session_manager,
                "query": shared.get("user_query", ""),
                "request_id": shared.get("request_id"),
            }

        async def exec_async(self, prep_res: dict[str, Any]) -> Any:
            workflow = prep_res.get("workflow")
            if not workflow:
                return None

            executor = WorkflowExecutor(
                session_manager=prep_res.get("session_manager"),
                emitter=prep_res.get("emitter"),
                request_id=prep_res.get("request_id"),
            )

            result = await executor.execute(
                workflow,
                initial_context={"user_query": prep_res["query"]},
            )

            return result

        async def post_async(
            self,
            shared: dict[str, Any],
            prep_res: dict[str, Any],
            exec_res: Any,
        ) -> str | None:
            if exec_res:
                shared.setdefault("results", {})
                shared["results"]["workflow_output"] = exec_res
            return "default"

    # Create nodes
    parse_workflow = ParseWorkflowNode()
    execute_workflow = ExecuteWorkflowNode()

    # Wire
    parse_workflow >> execute_workflow

    return AsyncFlow(start=parse_workflow)
