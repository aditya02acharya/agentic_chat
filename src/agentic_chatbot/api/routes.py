"""API routes."""

import asyncio
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from starlette.responses import StreamingResponse

from .models import ChatRequest, HealthResponse, ToolInfo, ToolsResponse
from .sse import event_stream
from .dependencies import get_app, get_settings_dep
from ..core.request_context import RequestContext
from ..flows.main_flow import execute_chat_flow
from ..operators.registry import OperatorRegistry
from ..operators.llm import QueryRewriterOperator, SynthesizerOperator, WriterOperator, AnalyzerOperator
from ..operators.mcp import RAGRetrieverOperator, WebSearcherOperator
from ..operators.hybrid import CoderOperator
from ..config.settings import Settings
from .. import __version__
from ..utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/chat")
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings_dep),
):
    """
    Main chat endpoint with SSE streaming.

    Returns Server-Sent Events stream with progress updates
    and response chunks.
    """
    ctx = RequestContext(
        conversation_id=request.conversation_id,
        user_query=request.message,
    )

    shared = {}
    if request.context:
        shared.update(request.context)

    try:
        app = get_app()
        if app.mcp_registry:
            shared["mcp_registry"] = app.mcp_registry
        if app.mcp_session_manager:
            shared["mcp_session_manager"] = app.mcp_session_manager
    except RuntimeError:
        pass

    async def run_and_stream():
        task = asyncio.create_task(execute_chat_flow(ctx, shared))

        try:
            async for event_data in event_stream(ctx):
                yield event_data
        except asyncio.CancelledError:
            task.cancel()
            raise
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await ctx.cleanup()

    return StreamingResponse(
        run_and_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-ID": ctx.request_id,
        },
    )


@router.get("/chat/{conversation_id}/history")
async def get_history(conversation_id: str):
    """Get conversation history."""
    return {"messages": [], "conversation_id": conversation_id}


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", version=__version__)


@router.get("/tools", response_model=ToolsResponse)
async def list_tools():
    """List available tools/operators."""
    operators = OperatorRegistry.list_operators()
    tools = [
        ToolInfo(
            name=op["name"],
            description=op["description"],
            type=op["type"],
        )
        for op in operators
    ]
    return ToolsResponse(tools=tools)
