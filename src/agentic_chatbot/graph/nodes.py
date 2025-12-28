"""LangGraph node functions for the agentic chatbot.

Nodes are async functions that:
- Take the current state as input
- Return a partial state dict with updates
- Can emit events for SSE streaming

Each node focuses on a single responsibility following the original design.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from langchain_core.messages import HumanMessage, AIMessage

from agentic_chatbot.config.prompts import (
    SUPERVISOR_SYSTEM_PROMPT,
    SUPERVISOR_DECISION_PROMPT,
    REFLECT_SYSTEM_PROMPT,
    REFLECT_PROMPT,
    SYNTHESIZER_PROMPT,
    WRITER_PROMPT,
    BLOCKED_HANDLER_PROMPT,
    WORKFLOW_PLANNER_SYSTEM_PROMPT,
    WORKFLOW_PLANNER_PROMPT,
)
from agentic_chatbot.events.models import (
    SupervisorThinkingEvent,
    SupervisorDecidedEvent,
    ToolStartEvent,
    ToolCompleteEvent,
    ToolErrorEvent,
    ResponseChunkEvent,
    ResponseDoneEvent,
    ClarifyRequestEvent,
    WorkflowCreatedEvent,
    WorkflowStepStartEvent,
    WorkflowStepCompleteEvent,
    WorkflowCompleteEvent,
)
from agentic_chatbot.graph.state import (
    ChatState,
    SupervisorDecision,
    ToolResult,
    ReflectionResult,
    ActionRecord,
    get_emitter,
    generate_source_id,
    get_summaries_text,
    get_sourced_contents,
    get_citation_blocks,
    get_document_summaries_text,
)
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.operators.context import OperatorContext, MessagingContext
from agentic_chatbot.utils.structured_llm import StructuredLLMCaller, StructuredOutputError
from agentic_chatbot.utils.llm import LLMClient
from agentic_chatbot.utils.logging import get_logger
from agentic_chatbot.config.models import TokenUsage

# New unified data model
from agentic_chatbot.data.content import ContentBlock, TextContent
from agentic_chatbot.data.sourced import SourcedContent, ContentSource, ContentSummary, create_sourced_content
from agentic_chatbot.data.execution import ExecutionInput, ExecutionOutput, ExecutionStatus, TaskInfo
from agentic_chatbot.core.workflow import (
    WorkflowDefinition,
    WorkflowDefinitionSchema,
    WorkflowResult,
    WorkflowStatus,
)
from agentic_chatbot.core.workflow_executor import WorkflowExecutor
from agentic_chatbot.events.emitter import EventEmitter


logger = get_logger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def emit_event(state: ChatState, event: Any) -> None:
    """Emit an event to the SSE stream."""
    emitter = get_emitter(state)
    if emitter:
        await emitter.emit(event)


def format_conversation_context(state: ChatState) -> str:
    """Format conversation history for prompts."""
    messages = state.get("messages", [])
    if not messages:
        return "No previous conversation."

    formatted = []
    for msg in messages[-10:]:  # Last 10 messages
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = msg.content if hasattr(msg, "content") else str(msg)
        formatted.append(f"{role}: {content}")

    return "\n".join(formatted)


def format_tool_results(state: ChatState) -> str:
    """Format tool results for prompts."""
    results = state.get("tool_results", [])
    if not results:
        return "No tool results yet."

    formatted = []
    for result in results:
        status = "✓" if result.success else "✗"
        formatted.append(f"{status} {result.tool_name}: {result.content[:200]}...")

    return "\n".join(formatted)


def get_tool_summaries(state: ChatState) -> str:
    """
    Get formatted tool summaries.

    Priority:
    1. UnifiedToolProvider (includes both local and remote tools)
    2. MCP registry (remote tools only)
    3. Operator registry (fallback)
    """
    # Prefer UnifiedToolProvider - includes local tools for self-awareness
    tool_provider = state.get("tool_provider")
    if tool_provider and hasattr(tool_provider, "get_tools_text"):
        return tool_provider.get_tools_text()

    # Fallback to MCP registry
    registry = state.get("mcp_registry")
    if registry and hasattr(registry, "get_tool_summaries_text"):
        return registry.get_tool_summaries_text()

    return OperatorRegistry.get_operators_text()


# =============================================================================
# INITIALIZATION NODE
# =============================================================================


async def initialize_node(state: ChatState) -> dict[str, Any]:
    """
    Initialize the chat session.

    - Validates input
    - Sets up initial state values
    - Adds user message to conversation history
    """
    logger.info("Initializing chat session", request_id=state.get("request_id"))

    user_query = state.get("user_query", "")
    if not user_query.strip():
        return {
            "error": "Empty user query",
            "error_type": "validation",
        }

    # Add user message to conversation
    user_message = HumanMessage(content=user_query)

    return {
        "messages": [user_message],
        "iteration": 0,
    }


# =============================================================================
# SUPERVISOR NODE
# =============================================================================


async def supervisor_node(state: ChatState) -> dict[str, Any]:
    """
    Central decision-making agent using ReACT pattern.

    CONTEXT OPTIMIZATION:
    - Uses SUMMARIES of tool outputs, not raw data (saves context)
    - Conversation history for continuity
    - Action history for tracking attempts
    - Delegates tasks with TaskContext (reformulated task, goal, scope)

    Decides:
    - ANSWER: Direct response (has enough info)
    - CALL_TOOL: Use a single tool (provides task_description for operator)
    - CREATE_WORKFLOW: Multi-step execution
    - CLARIFY: Ask for clarification
    """
    await emit_event(
        state,
        SupervisorThinkingEvent.create(
            "Analyzing your question...",
            request_id=state.get("request_id"),
        ),
    )

    # Check iteration limit
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 5)

    if iteration >= max_iterations:
        logger.warning("Max iterations reached", iteration=iteration)
        return {
            "current_decision": SupervisorDecision(
                action="ANSWER",
                reasoning="Maximum iterations reached, providing best available answer",
                response="I've gathered the available information. Let me summarize what I found.",
            ),
            "iteration": iteration + 1,
        }

    # Build OPTIMIZED context - summaries only, not raw data
    tool_summaries = get_tool_summaries(state)
    conversation_context = format_conversation_context(state)

    # Use data summaries instead of raw tool_results for decision making
    # This keeps supervisor context lean while having all decision-relevant info
    data_summaries_text = get_summaries_text(state)

    # Get document summaries (if any documents are uploaded)
    document_context = get_document_summaries_text(state)

    # Format action history
    action_history = state.get("action_history", [])
    actions_text = (
        "\n".join(f"- {a.action}: {a.reasoning[:50]}..." for a in action_history)
        or "None"
    )

    # Build prompts with summaries
    system = SUPERVISOR_SYSTEM_PROMPT.format(tool_summaries=tool_summaries)
    prompt = SUPERVISOR_DECISION_PROMPT.format(
        query=state.get("user_query", ""),
        conversation_context=conversation_context,
        document_context=document_context,  # Document summaries
        tool_results=data_summaries_text,  # Summaries, not raw data
        actions_this_turn=actions_text,
    )

    caller = StructuredLLMCaller(max_retries=3)

    try:
        # Use extended thinking for complex reasoning decisions
        result = await caller.call_with_usage(
            prompt=prompt,
            response_model=SupervisorDecision,
            system=system,
            model="thinking",
            enable_thinking=True,
            thinking_budget=10000,
        )

        decision = result.data
        token_usage = result.usage

        # Emit decided event
        action_messages = {
            "ANSWER": "I can answer this directly",
            "CALL_TOOL": f"Using {decision.operator or 'tool'} to get information",
            "CREATE_WORKFLOW": "Creating a multi-step plan",
            "CLARIFY": "I need some clarification",
        }
        await emit_event(
            state,
            SupervisorDecidedEvent.create(
                action=decision.action,
                message=action_messages.get(decision.action, "Processing..."),
                request_id=state.get("request_id"),
            ),
        )

        logger.info(
            "Supervisor decided",
            action=decision.action,
            thinking_tokens=token_usage.thinking_tokens,
        )

        # Record action
        action_record = ActionRecord(
            action=decision.action,
            reasoning=decision.reasoning,
            iteration=iteration,
        )

        # Create task info for operators (focused, not full conversation)
        task_info = decision.to_task_info(state.get("user_query", ""))

        return {
            "current_decision": decision,
            "current_tool": decision.operator,
            "current_params": decision.params,
            "workflow_goal": decision.goal,
            "iteration": iteration + 1,
            "action_history": [action_record],
            "current_task": task_info,  # For operators (unified data model)
            "token_usage": token_usage,  # Track token usage
        }

    except StructuredOutputError as e:
        logger.error(f"Supervisor decision failed: {e}")
        return {
            "current_decision": SupervisorDecision(
                action="CLARIFY",
                reasoning="Unable to determine appropriate action",
                question="Could you please rephrase your request?",
            ),
            "iteration": iteration + 1,
        }


# =============================================================================
# TOOL EXECUTION NODE
# =============================================================================


async def execute_tool_node(state: ChatState) -> dict[str, Any]:
    """
    Execute a single tool/operator with context optimization and messaging support.

    TOOL TYPES:
    - Local tools: Zero-latency, in-process (self_info, list_capabilities, etc.)
    - Remote MCP tools: Network calls to external servers
    - Operators: High-level execution strategies (may use MCP tools internally)

    CONTEXT OPTIMIZATION:
    - Receives TaskContext from supervisor (focused task, not full conversation)
    - Creates DataChunk with source tracking (for citations)
    - Generates inline summary (using haiku for speed)
    - Returns both raw data (for synthesizer) and summary (for supervisor)

    MESSAGING CAPABILITIES:
    - Wires MessagingContext for operators that support:
      - Progress updates
      - Direct responses (bypass writer)
      - User elicitation
      - Rich content (images, widgets)

    Flow:
    1. Check if local tool (execute via UnifiedToolProvider) or operator
    2. Execute tool with TaskContext and MessagingContext
    3. Store raw output as DataChunk (for synthesizer/writer citations)
    4. Generate inline summary (for supervisor decisions)
    5. Track if operator sent direct response (skip writer if so)
    """
    decision = state.get("current_decision")
    if not decision or not decision.operator:
        return {
            "tool_results": [
                ToolResult(
                    tool_name="unknown",
                    success=False,
                    error="No operator specified in decision",
                )
            ],
        }

    tool_name = decision.operator
    params = decision.params or {}
    task_info = state.get("current_task")
    tool_provider = state.get("tool_provider")

    await emit_event(
        state,
        ToolStartEvent.create(
            tool=tool_name,
            message=f"Executing {tool_name}...",
            request_id=state.get("request_id"),
        ),
    )

    # Check if this is a local tool (zero-latency, in-process)
    is_local = False
    if tool_provider and hasattr(tool_provider, "is_local_tool"):
        is_local = tool_provider.is_local_tool(tool_name)

    if is_local:
        # Execute local tool via UnifiedToolProvider
        return await _execute_local_tool(state, tool_name, params, task_info, tool_provider)

    # Otherwise, execute as operator (may use MCP tools internally)
    return await _execute_operator(state, tool_name, params, task_info)


async def _execute_local_tool(
    state: ChatState,
    tool_name: str,
    params: dict[str, Any],
    task_info: TaskInfo | None,
    tool_provider: Any,
) -> dict[str, Any]:
    """
    Execute a local tool via UnifiedToolProvider.

    Local tools are zero-latency, in-process tools like:
    - self_info: Bot version and capabilities
    - list_capabilities: Detailed feature list
    - list_tools: Available tools
    - list_operators: Available operators

    Returns SourcedContent with the tool output.
    """
    try:
        # Execute via tool provider
        result = await tool_provider.execute(
            tool_name,
            params,
            request_id=state.get("request_id"),
            conversation_id=state.get("conversation_id"),
        )

        await emit_event(
            state,
            ToolCompleteEvent.create(
                tool=tool_name,
                content_count=len(result.contents) if result.contents else 1,
                request_id=state.get("request_id"),
            ),
        )

        # Generate source ID for citations
        source_id, updated_counter = generate_source_id(state, tool_name)

        # Extract content from local tool result
        result_content = ""
        if result.contents:
            # Join all content items
            content_parts = []
            for content in result.contents:
                if hasattr(content, "text"):
                    content_parts.append(content.text)
                elif hasattr(content, "content"):
                    content_parts.append(str(content.content))
                else:
                    content_parts.append(str(content))
            result_content = "\n".join(content_parts)

        result_success = result.status.value == "success"
        result_error = result.error

        # Create SourcedContent (unified data model)
        query_used = task_info.description if task_info else state.get("user_query", "")
        sourced = create_sourced_content(
            content=result_content if result_success else f"Error: {result_error}",
            source_type=tool_name,
            source_id=source_id,
            query_used=query_used,
        )

        # Generate summary for supervisor (inline, using haiku)
        if result_success:
            summary = ContentSummary(
                executive_summary=f"Local tool {tool_name} returned data",
                key_findings=[result_content[:100] + "..." if len(result_content) > 100 else result_content],
                has_useful_data=bool(result_content),
                task_description=query_used,
            )
        else:
            summary = ContentSummary(
                executive_summary=f"Local tool {tool_name} failed",
                key_findings=[],
                has_useful_data=False,
                error=result_error,
                task_description=query_used,
            )
        sourced.set_summary(summary)

        # Create ExecutionOutput (new unified output type)
        if result_success:
            exec_output = ExecutionOutput.success(
                TextContent.markdown(result_content),
                sourced_contents=[sourced],
            )
        else:
            exec_output = ExecutionOutput.error(result_error or "Unknown error")

        logger.info(
            "Local tool executed",
            tool=tool_name,
            source_id=source_id,
            success=result_success,
        )

        return {
            "tool_results": [
                ToolResult(
                    tool_name=tool_name,
                    success=result_success,
                    content=result_content,
                    error=result_error,
                    metadata={"local_tool": True},
                )
            ],
            "current_tool": None,
            "sourced_contents": [sourced],
            "execution_outputs": [exec_output],
            "source_counter": updated_counter,
        }

    except Exception as e:
        logger.error(f"Local tool execution failed: {e}", exc_info=True)

        await emit_event(
            state,
            ToolErrorEvent.create(
                tool=tool_name,
                error=str(e),
                request_id=state.get("request_id"),
            ),
        )

        source_id, updated_counter = generate_source_id(state, tool_name)

        # Create error SourcedContent
        sourced = create_sourced_content(
            content=f"Error: {e}",
            source_type=tool_name,
            source_id=source_id,
            query_used=state.get("user_query", ""),
        )
        sourced.set_summary(ContentSummary(
            executive_summary="Local tool execution failed",
            key_findings=[],
            has_useful_data=False,
            error=str(e),
        ))

        return {
            "tool_results": [
                ToolResult(
                    tool_name=tool_name,
                    success=False,
                    error=str(e),
                    metadata={"local_tool": True},
                )
            ],
            "current_tool": None,
            "sourced_contents": [sourced],
            "execution_outputs": [ExecutionOutput.error(str(e))],
            "source_counter": updated_counter,
        }


async def _execute_operator(
    state: ChatState,
    tool_name: str,
    params: dict[str, Any],
    task_info: TaskInfo | None,
) -> dict[str, Any]:
    """
    Execute an operator with messaging context support.

    Operators are high-level execution strategies that may:
    - Use MCP tools internally
    - Send direct responses to user
    - Support progress updates and elicitation

    Returns SourcedContent with the operator output.
    """
    try:
        # Get operator from registry
        operator_cls = OperatorRegistry.get(tool_name)
        if not operator_cls:
            raise ValueError(f"Operator {tool_name} not found")

        operator = operator_cls()

        # Create MessagingContext for operators that support messaging capabilities
        emitter = get_emitter(state)
        elicitation_manager = state.get("elicitation_manager")

        messaging_context = MessagingContext(
            emitter=emitter,
            elicitation_manager=elicitation_manager,
            request_id=state.get("request_id"),
            operator_name=tool_name,
        )

        # Create operator context with TaskInfo (focused, not full conversation)
        query = task_info.description if task_info else state.get("user_query", "")

        context = OperatorContext(
            query=query,
            recent_messages=[],
            conversation_summary="",
            tool_schemas={},
            step_results={},
            extra={
                "params": params,
                "task_goal": task_info.goal if task_info else "",
                "task_scope": task_info.scope if task_info else "",
                "conversation_id": state.get("conversation_id", ""),
                "request_id": state.get("request_id", ""),
            },
            shared_store={},
        )

        # Wire the messaging context to the operator context
        context.set_messaging(messaging_context)

        # Execute operator
        result = await operator.execute(context)

        await emit_event(
            state,
            ToolCompleteEvent.create(
                tool=tool_name,
                content_count=1,
                request_id=state.get("request_id"),
            ),
        )

        # Generate unique source ID for citations
        source_id, updated_counter = generate_source_id(state, tool_name)

        # Get result content
        result_content = ""
        result_success = True
        result_error = None
        result_metadata = {}

        if hasattr(result, "output"):
            result_content = result.text_output if hasattr(result, "text_output") else str(result.output)
            result_success = result.success
            result_error = result.error
            result_metadata = result.metadata if hasattr(result, "metadata") else {}
        elif hasattr(result, "content"):
            result_content = result.content
            result_success = getattr(result, "success", True)
            result_error = getattr(result, "error", None)
            result_metadata = getattr(result, "metadata", {})

        # Create SourcedContent (unified data model)
        query_used = task_info.description if task_info else state.get("user_query", "")
        sourced = create_sourced_content(
            content=result_content if result_success else f"Error: {result_error}",
            source_type=tool_name,
            source_id=source_id,
            query_used=query_used,
        )

        # Create summary for supervisor
        if result_success:
            summary = ContentSummary(
                executive_summary=f"Operator {tool_name} completed successfully",
                key_findings=[result_content[:150] + "..." if len(result_content) > 150 else result_content] if result_content else [],
                has_useful_data=bool(result_content),
                task_description=query_used,
            )
        else:
            summary = ContentSummary(
                executive_summary=f"Operator {tool_name} failed",
                key_findings=[],
                has_useful_data=False,
                error=result_error,
                task_description=query_used,
            )
        sourced.set_summary(summary)

        # Check for direct responses
        sent_direct_response = False
        direct_response_contents = []

        if hasattr(result, "sent_direct_response") and result.sent_direct_response:
            sent_direct_response = True
            if hasattr(result, "direct_responses"):
                direct_response_contents = result.direct_responses

        if messaging_context.has_direct_responses:
            sent_direct_response = True
            direct_response_contents.extend(messaging_context.direct_responses)

        # Create ExecutionOutput
        if result_success:
            exec_output = ExecutionOutput.success(
                TextContent.markdown(result_content),
                sourced_contents=[sourced],
                sent_direct_response=sent_direct_response,
                input_tokens=result.input_tokens if hasattr(result, "input_tokens") else 0,
                output_tokens=result.output_tokens if hasattr(result, "output_tokens") else 0,
            )
        else:
            exec_output = ExecutionOutput.error(result_error or "Unknown error")

        logger.info(
            "Operator executed",
            tool=tool_name,
            source_id=source_id,
            has_summary=bool(summary.key_findings),
            sent_direct_response=sent_direct_response,
        )

        return {
            "tool_results": [
                ToolResult(
                    tool_name=tool_name,
                    success=result_success,
                    content=result_content,
                    error=result_error,
                    metadata=result_metadata,
                )
            ],
            "current_tool": None,
            "sourced_contents": [sourced],
            "execution_outputs": [exec_output],
            "source_counter": updated_counter,
            "sent_direct_response": sent_direct_response,
            "direct_response_contents": direct_response_contents,
        }

    except Exception as e:
        logger.error(f"Tool execution failed: {e}", exc_info=True)

        await emit_event(
            state,
            ToolErrorEvent.create(
                tool=tool_name,
                error=str(e),
                request_id=state.get("request_id"),
            ),
        )

        source_id, updated_counter = generate_source_id(state, tool_name)

        # Create error SourcedContent
        sourced = create_sourced_content(
            content=f"Error: {e}",
            source_type=tool_name,
            source_id=source_id,
            query_used=state.get("user_query", ""),
        )
        sourced.set_summary(ContentSummary(
            executive_summary="Tool execution failed",
            key_findings=[],
            has_useful_data=False,
            error=str(e),
        ))

        return {
            "tool_results": [
                ToolResult(
                    tool_name=tool_name,
                    success=False,
                    error=str(e),
                )
            ],
            "current_tool": None,
            "sourced_contents": [sourced],
            "execution_outputs": [ExecutionOutput.error(str(e))],
            "source_counter": updated_counter,
        }


# =============================================================================
# WORKFLOW PLANNING NODE
# =============================================================================


async def plan_workflow_node(state: ChatState) -> dict[str, Any]:
    """
    Plan a multi-step workflow for complex tasks.

    Uses the supervisor's goal and hints to create a detailed
    WorkflowDefinition with steps, dependencies, and input mappings.

    This enables:
    - Multi-step execution with dependency handling
    - Parallel execution of independent steps
    - Template-based input resolution between steps
    """
    decision = state.get("current_decision")
    user_query = state.get("user_query", "")

    # Get goal from supervisor decision
    goal = decision.goal if decision else user_query
    supervisor_hints = decision.steps if decision and decision.steps else []

    await emit_event(
        state,
        SupervisorThinkingEvent.create(
            "Creating execution plan...",
            request_id=state.get("request_id"),
        ),
    )

    logger.info("Planning workflow", goal=goal)

    # Build context from existing results and conversation
    context_parts = []
    context_parts.append(f"User Query: {user_query}")

    if supervisor_hints:
        context_parts.append(f"Supervisor hints: {supervisor_hints}")

    tool_results = state.get("tool_results", [])
    if tool_results:
        results_text = format_tool_results(state)
        context_parts.append(f"Available context:\n{results_text}")

    context = "\n\n".join(context_parts)

    # Get operator summaries for the planner
    operator_summaries = get_tool_summaries(state)

    # Build prompts
    system = WORKFLOW_PLANNER_SYSTEM_PROMPT.format(operator_summaries=operator_summaries)
    prompt = WORKFLOW_PLANNER_PROMPT.format(
        query=goal,
        context=context,
    )

    caller = StructuredLLMCaller(max_retries=3)

    try:
        # Use extended thinking for complex workflow planning
        result = await caller.call_with_usage(
            prompt=prompt,
            response_model=WorkflowDefinitionSchema,
            system=system,
            model="thinking",
            enable_thinking=True,
            thinking_budget=15000,  # More budget for complex planning
        )

        workflow_schema = result.data
        token_usage = result.usage

        # Convert to runtime WorkflowDefinition
        workflow = WorkflowDefinition.from_schema(workflow_schema)

        # Emit workflow created event
        await emit_event(
            state,
            WorkflowCreatedEvent.create(
                goal=workflow.goal,
                steps=len(workflow.steps),
                request_id=state.get("request_id"),
            ),
        )

        logger.info(
            "Workflow planned",
            goal=workflow.goal,
            steps=len(workflow.steps),
            step_names=[s.name for s in workflow.steps],
            thinking_tokens=token_usage.thinking_tokens,
        )

        # Store workflow definition in state for execution
        return {
            "workflow_definition": workflow,
            "workflow_goal": workflow.goal,
            "token_usage": token_usage,  # Track token usage
        }

    except StructuredOutputError as e:
        logger.error(f"Workflow planning failed: {e}")
        # Fall back to treating as single tool call
        return {
            "workflow_definition": None,
            "error": f"Failed to plan workflow: {e}",
            "error_type": "planning",
        }


# =============================================================================
# WORKFLOW EXECUTION NODE
# =============================================================================


async def execute_workflow_node(state: ChatState) -> dict[str, Any]:
    """
    Execute a planned workflow using the WorkflowExecutor.

    Features:
    - Parallel execution of independent steps (via topological batching)
    - Dependency resolution with input template substitution
    - Per-step event emission for progress tracking
    - Error handling with partial result collection

    The workflow executor uses a DAG approach:
    1. Group steps into batches based on dependencies
    2. Execute each batch (parallel within batch)
    3. Pass outputs to dependent steps via input_mapping templates
    """
    workflow = state.get("workflow_definition")

    if not workflow:
        logger.warning("No workflow definition found, skipping execution")
        return {
            "tool_results": [
                ToolResult(
                    tool_name="workflow",
                    success=False,
                    error="No workflow definition to execute",
                )
            ],
        }

    # Create event emitter
    emitter = get_emitter(state)

    # Create workflow executor with MCP session manager if available
    executor = WorkflowExecutor(
        session_manager=state.get("mcp_session_manager"),
        emitter=emitter,
        request_id=state.get("request_id"),
    )

    logger.info(
        "Executing workflow",
        goal=workflow.goal,
        steps=len(workflow.steps),
    )

    try:
        # Execute the workflow with parallel optimization
        result: WorkflowResult = await executor.execute(
            workflow,
            initial_context={
                "user_query": state.get("user_query", ""),
                "conversation_id": state.get("conversation_id", ""),
            },
        )

        # Convert workflow results to tool results for reflection
        tool_results = []
        for step_id, step_result in result.steps.items():
            tool_results.append(
                ToolResult(
                    tool_name=f"workflow.{step_id}",
                    success=step_result.status == WorkflowStatus.COMPLETED,
                    content=str(step_result.output) if step_result.output else "",
                    error=step_result.error,
                    metadata={
                        "step_id": step_id,
                        "duration_ms": step_result.duration_ms,
                    },
                )
            )

        logger.info(
            "Workflow execution complete",
            goal=workflow.goal,
            status=result.status.value,
            failed_steps=result.failed_steps,
        )

        return {
            "tool_results": tool_results,
            "workflow_result": result,
            "workflow_completed": result.is_complete,
        }

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)

        # Emit error event
        await emit_event(
            state,
            WorkflowCompleteEvent.create(
                request_id=state.get("request_id"),
            ),
        )

        return {
            "tool_results": [
                ToolResult(
                    tool_name="workflow",
                    success=False,
                    error=str(e),
                )
            ],
            "workflow_completed": False,
        }


# =============================================================================
# REFLECTION NODE
# =============================================================================


async def reflect_node(state: ChatState) -> dict[str, Any]:
    """
    Analyze results and decide next action.

    CONTEXT OPTIMIZATION:
    - Uses SUMMARIES for reflection (not raw data)
    - Fast assessment using haiku
    - Summaries contain key findings relevant to the task

    Returns assessment:
    - satisfied: Have enough info to answer
    - need_more: Need additional tools/info
    - blocked: Cannot proceed
    """
    user_query = state.get("user_query", "")

    # Use summaries for reflection (context optimization)
    summaries_text = get_summaries_text(state)

    system = REFLECT_SYSTEM_PROMPT
    prompt = REFLECT_PROMPT.format(
        query=user_query,
        tool_results=summaries_text,  # Summaries, not raw data
        iteration=state.get("iteration", 0),
        max_iterations=state.get("max_iterations", 5),
    )

    caller = StructuredLLMCaller(max_retries=2)

    try:
        reflection = await caller.call(
            prompt=prompt,
            response_model=ReflectionResult,
            system=system,
            model="haiku",
        )

        logger.info("Reflection complete", assessment=reflection.assessment)

        return {"reflection": reflection}

    except StructuredOutputError:
        # Default to satisfied if reflection fails
        return {
            "reflection": ReflectionResult(
                assessment="satisfied",
                reasoning="Proceeding with available information",
            )
        }


# =============================================================================
# SYNTHESIS NODE
# =============================================================================


async def synthesize_node(state: ChatState) -> dict[str, Any]:
    """
    Synthesize data into a coherent response with citation support.

    Uses SourcedContent (unified data model):
    - SourcedContent.content for full data (verbatim for accuracy)
    - SourcedContent.source for citation tracking
    - Produces content that writer can cite using [^source_id] markers

    The synthesizer receives full data context, not summaries,
    to ensure accurate information in the final response.
    """
    user_query = state.get("user_query", "")

    # Get sourced contents (unified data model)
    sourced_contents = get_sourced_contents(state)

    if not sourced_contents:
        # Fallback to tool_results if no sourced contents
        tool_results = state.get("tool_results", [])
        results_text = "\n\n".join(
            f"**{r.tool_name}**:\n{r.content}"
            for r in tool_results
            if r.success
        )
    else:
        # Format sourced contents with source IDs for citation tracking
        results_text = "\n\n".join(
            f"[Source: {sc.source_id}] ({sc.source_type}):\n{sc.text}"
            for sc in sourced_contents
            if sc.text and not sc.text.startswith("Error:")
        )

    # Build citation reference block for the writer
    citation_block = ""
    if sourced_contents:
        citation_block = "\n\n---\nAvailable Sources for Citations:\n"
        for sc in sourced_contents:
            citation_block += f"- [^{sc.source_id}]: {sc.source_type}\n"

    prompt = SYNTHESIZER_PROMPT.format(
        query=user_query,
        tool_results=results_text + citation_block,
    )

    client = LLMClient()
    response = await client.generate(prompt, model="sonnet")

    return {"final_response": response}


# =============================================================================
# WRITE NODE
# =============================================================================


async def write_node(state: ChatState) -> dict[str, Any]:
    """
    Format the final response with GitHub footnote citations.

    Uses SourcedContent (unified data model):
    - SourcedContent.content for full data
    - SourcedContent.source for citation tracking
    - Adds GitHub-style footnotes [^source_id]

    MODEL SELECTION:
    - Uses requested_model from state if provided by user
    - Falls back to default 'sonnet' model

    Citation Format (GitHub Footnotes):
    - In text: "The data shows X[^web_search_1]"
    - At end: "[^web_search_1]: Source: web_search"
    """
    decision = state.get("current_decision")
    existing_response = state.get("final_response", "")
    sourced_contents = get_sourced_contents(state)

    # Get requested model from state (user preference) or use default
    requested_model = state.get("requested_model") or "sonnet"

    # Track token usage for writer (only set if LLM is called)
    writer_token_usage = None

    # If we already have a synthesized response, use it
    if existing_response:
        final = existing_response
    elif decision and decision.action == "ANSWER" and decision.response:
        final = decision.response
    else:
        # Generate response with citation context
        if sourced_contents:
            context_parts = []
            for sc in sourced_contents:
                if sc.text and not sc.text.startswith("Error:"):
                    context_parts.append(
                        f"[^{sc.source_id}] ({sc.source_type}):\n{sc.text}"
                    )
            context = "\n\n".join(context_parts)
        else:
            context = format_tool_results(state)

        prompt = WRITER_PROMPT.format(
            query=state.get("user_query", ""),
            context=context,
        )
        client = LLMClient()
        response = await client.complete(prompt, model=requested_model)
        final = response.content

        # Track token usage for writer
        writer_token_usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        logger.info(
            "Writer generated response",
            model=requested_model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    # Add footnote references if we have sourced contents and citations were used
    if sourced_contents and "[^" in final:
        footnotes = "\n\n---\n"
        for sc in sourced_contents:
            # Only add footnotes for sources that are actually cited
            if f"[^{sc.source_id}]" in final:
                footnotes += f"\n[^{sc.source_id}]: {sc.source_type}"
                if sc.source.query_used:
                    footnotes += f" (query: \"{sc.source.query_used[:50]}...\")"

        # Only append if we have any footnotes
        if footnotes != "\n\n---\n":
            final = final + footnotes

    # Add assistant message to conversation
    assistant_message = AIMessage(content=final)

    result = {
        "final_response": final,
        "messages": [assistant_message],
    }

    # Include token usage if LLM was called
    if writer_token_usage:
        result["token_usage"] = writer_token_usage

    return result


# =============================================================================
# STREAM NODE
# =============================================================================


async def stream_node(state: ChatState) -> dict[str, Any]:
    """
    Stream the final response to the user via SSE.

    Handles two scenarios:
    1. Normal flow: Stream the final_response in chunks
    2. Direct response: Operator already sent content directly, just emit completion

    Emits response chunks and completion event.
    """
    request_id = state.get("request_id")

    # Check if direct response was already sent
    if state.get("sent_direct_response"):
        # Operator already sent content directly to user
        # Just emit completion event
        logger.info(
            "Direct response was sent by operator, skipping response streaming",
            request_id=request_id,
        )
        await emit_event(
            state,
            ResponseDoneEvent.create(request_id=request_id),
        )
        return {"response_chunks": ["[Direct response sent by operator]"]}

    # Normal flow: stream the final response
    response = state.get("final_response", "")

    # Stream response in chunks
    chunk_size = 50
    for i in range(0, len(response), chunk_size):
        chunk = response[i : i + chunk_size]
        await emit_event(
            state,
            ResponseChunkEvent.create(content=chunk, request_id=request_id),
        )

    # Emit completion event
    await emit_event(
        state,
        ResponseDoneEvent.create(request_id=request_id),
    )

    return {"response_chunks": [response]}


# =============================================================================
# CLARIFY NODE
# =============================================================================


async def clarify_node(state: ChatState) -> dict[str, Any]:
    """
    Ask the user for clarification.

    Emits clarification request event.
    """
    decision = state.get("current_decision")
    question = decision.question if decision else "Could you please provide more details?"

    await emit_event(
        state,
        ClarifyRequestEvent.create(
            question=question,
            request_id=state.get("request_id"),
        ),
    )

    return {
        "clarify_question": question,
        "final_response": question,
    }


# =============================================================================
# BLOCKED HANDLER NODE
# =============================================================================


async def handle_blocked_node(state: ChatState) -> dict[str, Any]:
    """
    Handle cases where the agent cannot proceed.

    Generates a graceful response explaining the limitation.
    """
    reflection = state.get("reflection")
    reason = reflection.reasoning if reflection else "Unable to complete the request"

    prompt = BLOCKED_HANDLER_PROMPT.format(
        query=state.get("user_query", ""),
        reason=reason,
        attempts=format_tool_results(state),
    )

    client = LLMClient()
    response = await client.generate(prompt, model="sonnet")

    return {"final_response": response}


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================


def route_supervisor_decision(state: ChatState) -> Literal["answer", "call_tool", "create_workflow", "clarify"]:
    """Route based on supervisor decision."""
    decision = state.get("current_decision")
    if not decision:
        return "clarify"

    action_map = {
        "ANSWER": "answer",
        "CALL_TOOL": "call_tool",
        "CREATE_WORKFLOW": "create_workflow",
        "CLARIFY": "clarify",
    }
    return action_map.get(decision.action, "clarify")


def route_reflection(state: ChatState) -> Literal["satisfied", "need_more", "blocked", "direct_response"]:
    """
    Route based on reflection result.

    If a direct response was already sent to the user (operator bypassed writer),
    route to 'direct_response' which skips synthesis and write nodes.
    """
    # Check if direct response was already sent
    if state.get("sent_direct_response"):
        return "direct_response"

    reflection = state.get("reflection")
    if not reflection:
        return "satisfied"

    return reflection.assessment
