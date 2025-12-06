"""FastAPI routes for the chat API."""

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from agentic_chatbot import __version__
from agentic_chatbot.api.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ToolsResponse,
    ToolSummaryResponse,
    ErrorResponse,
)
from agentic_chatbot.api.dependencies import (
    MCPRegistryDep,
    MCPSessionManagerDep,
)
from agentic_chatbot.api.sse import event_generator_with_task
from agentic_chatbot.events.models import Event, ErrorEvent, ResponseDoneEvent
from agentic_chatbot.flows.main_flow import create_main_chat_flow
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns application status and version.
    """
    return HealthResponse(
        status="ok",
        version=__version__,
    )


@router.get("/tools", response_model=ToolsResponse)
async def list_tools(
    mcp_registry: MCPRegistryDep,
) -> ToolsResponse:
    """
    List available tools.

    Returns summaries of all registered tools from both
    MCP servers and internal operators.
    """
    tools = []

    # Get MCP tools
    if mcp_registry:
        try:
            mcp_summaries = await mcp_registry.get_all_tool_summaries()
            for summary in mcp_summaries:
                tools.append(
                    ToolSummaryResponse(
                        name=summary.name,
                        description=summary.description,
                        server_id=summary.server_id,
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to get MCP tools: {e}")

    # Get operator summaries
    operator_summaries = OperatorRegistry.get_all_summaries()
    for summary in operator_summaries:
        # Avoid duplicates
        if not any(t.name == summary["name"] for t in tools):
            tools.append(
                ToolSummaryResponse(
                    name=summary["name"],
                    description=summary["description"],
                )
            )

    return ToolsResponse(tools=tools, count=len(tools))


@router.post("/chat")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    mcp_registry: MCPRegistryDep,
    mcp_session_manager: MCPSessionManagerDep,
) -> StreamingResponse:
    """
    Main chat endpoint with SSE streaming.

    Returns Server-Sent Events stream with progress updates
    and response chunks.
    """
    # Check if shutdown is in progress
    if getattr(request.app.state, "is_shutting_down", False):
        raise HTTPException(status_code=503, detail="Service is shutting down")

    # Create request context
    event_queue: asyncio.Queue[Event] = asyncio.Queue()
    request_id = f"{chat_request.conversation_id}_{asyncio.get_event_loop().time()}"

    # Build shared store
    shared: dict[str, Any] = {
        "user_query": chat_request.message,
        "conversation_id": chat_request.conversation_id,
        "request_id": request_id,
        "event_queue": event_queue,
        "mcp": {
            "server_registry": mcp_registry,
            "session_manager": mcp_session_manager,
        },
    }

    # Add context if provided
    if chat_request.context:
        shared["user_context"] = chat_request.context

    # Track request
    active_requests = getattr(request.app.state, "active_requests", set())
    active_requests.add(request_id)

    async def run_flow() -> None:
        """Run the chat flow in background."""
        try:
            flow = create_main_chat_flow()
            await flow.run_async(shared)
        except Exception as e:
            logger.error(f"Flow execution error: {e}", exc_info=True)
            await event_queue.put(
                ErrorEvent.create(
                    error=str(e),
                    error_type="flow_error",
                    request_id=request_id,
                )
            )
        finally:
            # Ensure done event is sent
            if event_queue.empty() or not any(
                isinstance(e, (ResponseDoneEvent, ErrorEvent))
                for e in list(event_queue._queue)  # type: ignore
            ):
                await event_queue.put(
                    ResponseDoneEvent.create(request_id=request_id)
                )
            # Untrack request
            active_requests.discard(request_id)

    # Start flow execution
    task = asyncio.create_task(run_flow())

    return StreamingResponse(
        event_generator_with_task(event_queue, task),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-ID": request_id,
        },
    )


@router.post("/chat/sync", response_model=ChatResponse)
async def chat_sync(
    request: Request,
    chat_request: ChatRequest,
    mcp_registry: MCPRegistryDep,
    mcp_session_manager: MCPSessionManagerDep,
) -> ChatResponse:
    """
    Non-streaming chat endpoint.

    Alternative to SSE streaming for simpler integrations.
    Waits for complete response before returning.
    """
    # Check if shutdown is in progress
    if getattr(request.app.state, "is_shutting_down", False):
        raise HTTPException(status_code=503, detail="Service is shutting down")

    request_id = f"{chat_request.conversation_id}_{asyncio.get_event_loop().time()}"

    # Build shared store
    shared: dict[str, Any] = {
        "user_query": chat_request.message,
        "conversation_id": chat_request.conversation_id,
        "request_id": request_id,
        "event_queue": asyncio.Queue(),
        "mcp": {
            "server_registry": mcp_registry,
            "session_manager": mcp_session_manager,
        },
    }

    try:
        flow = create_main_chat_flow()
        await flow.run_async(shared)

        # Get response from results
        results = shared.get("results", {})
        response = results.get("final_response", "")

        return ChatResponse(
            conversation_id=chat_request.conversation_id,
            response=response,
            request_id=request_id,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
