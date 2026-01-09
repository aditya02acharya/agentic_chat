"""FastAPI routes for the chat API."""

import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from agentic_chatbot import __version__
from agentic_chatbot.api.rate_limit import get_rate_limiter
from agentic_chatbot.api.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ToolsResponse,
    ToolSummaryResponse,
    TokenUsageResponse,
    ElicitationResponseRequest,
    ElicitationResponseResult,
    PendingElicitationResponse,
    PendingElicitationsResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
    DocumentStatusResponse,
    DocumentSummaryResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
)
from agentic_chatbot.api.dependencies import (
    MCPRegistryDep,
    MCPSessionManagerDep,
    ElicitationManagerDep,
    ToolProviderDep,
    DocumentServiceDep,
)
from agentic_chatbot.api.sse import event_generator_with_task
from agentic_chatbot.events.emitter import EventEmitter
from agentic_chatbot.events.models import Event, ErrorEvent, ResponseDoneEvent
from agentic_chatbot.graph import create_chat_graph
from agentic_chatbot.graph.state import create_initial_state
from agentic_chatbot.mcp.callbacks import create_mcp_callbacks
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)
router = APIRouter()

# Maximum time for graph execution (5 minutes)
GRAPH_EXECUTION_TIMEOUT = 300.0

# Background task tracking for graceful shutdown
_background_tasks: set[asyncio.Task] = set()


def _generate_request_id(conversation_id: str) -> str:
    """Generate a unique request ID using UUID4."""
    return f"{conversation_id}_{uuid.uuid4().hex[:12]}"


def _sanitize_error_message(error: Exception) -> str:
    """Sanitize error message for client response."""
    error_str = str(error).lower()

    if "connection" in error_str or "network" in error_str:
        return "A network error occurred. Please try again."
    elif "timeout" in error_str:
        return "The request timed out. Please try again."
    elif "not found" in error_str:
        return str(error)
    elif "limit" in error_str or "exceeded" in error_str:
        return str(error)
    elif "validation" in error_str or "invalid" in error_str:
        return str(error)
    elif "permission" in error_str or "unauthorized" in error_str:
        return "You don't have permission to perform this action."
    else:
        return "An unexpected error occurred. Please try again later."


def _track_background_task(task: asyncio.Task) -> None:
    """Track a background task for graceful shutdown."""
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def cleanup_background_tasks(timeout: float = 5.0) -> int:
    """Wait for background tasks to complete during shutdown."""
    if not _background_tasks:
        return 0

    logger.info(f"Waiting for {len(_background_tasks)} background tasks...")

    done, pending = await asyncio.wait(
        _background_tasks,
        timeout=timeout,
        return_when=asyncio.ALL_COMPLETED,
    )

    cancelled = 0
    for task in pending:
        task.cancel()
        cancelled += 1

    if cancelled:
        logger.warning(f"Cancelled {cancelled} background tasks that didn't complete")

    return cancelled


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok", version=__version__)


@router.get("/tools", response_model=ToolsResponse)
async def list_tools(tool_provider: ToolProviderDep) -> ToolsResponse:
    """List available tools."""
    tools = []

    try:
        all_summaries = await tool_provider.get_all_summaries()
        for summary in all_summaries:
            tools.append(
                ToolSummaryResponse(
                    name=summary.name,
                    description=summary.description,
                    server_id=getattr(summary, "server_id", None),
                )
            )
    except Exception as e:
        logger.warning(f"Failed to get tool summaries: {e}")

    operator_summaries = OperatorRegistry.get_all_summaries()
    for summary in operator_summaries:
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
    elicitation_manager: ElicitationManagerDep,
    tool_provider: ToolProviderDep,
) -> StreamingResponse:
    """
    Main chat endpoint with SSE streaming.

    Returns Server-Sent Events stream with progress updates and response chunks.
    """
    await get_rate_limiter().check(request)

    if getattr(request.app.state, "is_shutting_down", False):
        raise HTTPException(status_code=503, detail="Service is shutting down")

    event_queue: asyncio.Queue[Event] = asyncio.Queue()
    request_id = _generate_request_id(chat_request.conversation_id)
    emitter = EventEmitter(event_queue)

    mcp_callbacks = create_mcp_callbacks(
        emitter=emitter,
        elicitation_manager=elicitation_manager,
        request_id=request_id,
    )

    initial_state = create_initial_state(
        user_query=chat_request.message,
        conversation_id=chat_request.conversation_id,
        request_id=request_id,
        event_emitter=emitter,
        event_queue=event_queue,
        mcp_registry=mcp_registry,
        mcp_session_manager=mcp_session_manager,
        mcp_callbacks=mcp_callbacks,
        elicitation_manager=elicitation_manager,
        tool_provider=tool_provider,
        user_context=chat_request.context,
        requested_model=chat_request.model,
        user_id=chat_request.user_id,
    )

    active_requests = getattr(request.app.state, "active_requests", set())
    active_requests.add(request_id)

    async def run_graph() -> None:
        """Run the LangGraph in background with timeout protection."""
        try:
            graph = create_chat_graph()
            config = {"configurable": {"thread_id": chat_request.conversation_id}}

            await asyncio.wait_for(
                graph.ainvoke(initial_state, config),
                timeout=GRAPH_EXECUTION_TIMEOUT,
            )

        except asyncio.TimeoutError:
            logger.warning(f"Graph execution timeout after {GRAPH_EXECUTION_TIMEOUT}s")
            await event_queue.put(
                ErrorEvent.create(
                    error=f"Request timed out after {GRAPH_EXECUTION_TIMEOUT} seconds",
                    error_type="timeout",
                    request_id=request_id,
                )
            )
        except Exception as e:
            logger.error(f"Graph execution error: {e}", exc_info=True)
            await event_queue.put(
                ErrorEvent.create(
                    error=str(e),
                    error_type="graph_error",
                    request_id=request_id,
                )
            )
        finally:
            await event_queue.put(ResponseDoneEvent.create(request_id=request_id))
            active_requests.discard(request_id)

    task = asyncio.create_task(run_graph())
    _track_background_task(task)

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
    elicitation_manager: ElicitationManagerDep,
    tool_provider: ToolProviderDep,
) -> ChatResponse:
    """Non-streaming chat endpoint."""
    await get_rate_limiter().check(request)

    if getattr(request.app.state, "is_shutting_down", False):
        raise HTTPException(status_code=503, detail="Service is shutting down")

    request_id = _generate_request_id(chat_request.conversation_id)
    event_queue: asyncio.Queue[Event] = asyncio.Queue()
    emitter = EventEmitter(event_queue)

    mcp_callbacks = create_mcp_callbacks(
        emitter=emitter,
        elicitation_manager=elicitation_manager,
        request_id=request_id,
    )

    initial_state = create_initial_state(
        user_query=chat_request.message,
        conversation_id=chat_request.conversation_id,
        request_id=request_id,
        event_emitter=emitter,
        event_queue=event_queue,
        mcp_registry=mcp_registry,
        mcp_session_manager=mcp_session_manager,
        mcp_callbacks=mcp_callbacks,
        elicitation_manager=elicitation_manager,
        tool_provider=tool_provider,
        user_context=chat_request.context,
        requested_model=chat_request.model,
        user_id=chat_request.user_id,
    )

    try:
        graph = create_chat_graph()
        config = {"configurable": {"thread_id": chat_request.conversation_id}}

        try:
            final_state = await asyncio.wait_for(
                graph.ainvoke(initial_state, config),
                timeout=GRAPH_EXECUTION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Request timed out after {GRAPH_EXECUTION_TIMEOUT} seconds",
            )

        response = final_state.get("final_response", "")
        token_usage = final_state.get("token_usage")
        usage_response = None
        if token_usage:
            usage_response = TokenUsageResponse(
                user_input_tokens=token_usage.user_input_tokens,
                final_output_tokens=token_usage.final_output_tokens,
                intermediate_tokens=token_usage.intermediate_tokens,
                total_tokens=token_usage.total_tokens,
                input_tokens=token_usage.input_tokens,
                output_tokens=token_usage.output_tokens,
                thinking_tokens=token_usage.thinking_tokens,
                cache_read_tokens=token_usage.cache_read_tokens,
                cache_write_tokens=token_usage.cache_write_tokens,
            )

        return ChatResponse(
            conversation_id=chat_request.conversation_id,
            response=response,
            request_id=request_id,
            usage=usage_response,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_sanitize_error_message(e))


# =============================================================================
# ELICITATION ENDPOINTS
# =============================================================================


@router.post("/elicitation/respond", response_model=ElicitationResponseResult)
async def submit_elicitation_response(
    elicitation_request: ElicitationResponseRequest,
    elicitation_manager: ElicitationManagerDep,
) -> ElicitationResponseResult:
    """Submit user response to a pending elicitation request."""
    success = await elicitation_manager.submit_response(
        elicitation_id=elicitation_request.elicitation_id,
        value=elicitation_request.value,
        cancelled=elicitation_request.cancelled,
    )

    return ElicitationResponseResult(
        success=success,
        elicitation_id=elicitation_request.elicitation_id,
        message="Response accepted" if success else "Elicitation not found or already responded",
    )


@router.get("/elicitation/pending", response_model=PendingElicitationsResponse)
async def list_pending_elicitations(
    elicitation_manager: ElicitationManagerDep,
) -> PendingElicitationsResponse:
    """List all pending elicitation requests."""
    pending = elicitation_manager.get_all_pending()
    elicitations = [
        PendingElicitationResponse(
            elicitation_id=p.elicitation_id,
            server_id=p.server_id,
            tool_name=p.tool_name,
            prompt=p.request.prompt,
            input_type=p.request.input_type,
            options=p.request.options,
            default=p.request.default,
            timeout_seconds=p.request.timeout_seconds,
        )
        for p in pending
    ]
    return PendingElicitationsResponse(elicitations=elicitations, count=len(elicitations))


@router.delete("/elicitation/{elicitation_id}")
async def cancel_elicitation(
    elicitation_id: str,
    elicitation_manager: ElicitationManagerDep,
) -> ElicitationResponseResult:
    """Cancel a pending elicitation request."""
    success = await elicitation_manager.cancel_elicitation(elicitation_id)

    return ElicitationResponseResult(
        success=success,
        elicitation_id=elicitation_id,
        message="Elicitation cancelled" if success else "Elicitation not found or already responded",
    )


# =============================================================================
# DOCUMENT ENDPOINTS
# =============================================================================


@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    request: Request,
    upload_request: DocumentUploadRequest,
    document_service: DocumentServiceDep,
) -> DocumentUploadResponse:
    """Upload a document for conversation context."""
    await get_rate_limiter().check(request)

    if not document_service:
        raise HTTPException(status_code=503, detail="Document service not available")

    try:
        document_id = await document_service.create_document(
            conversation_id=upload_request.conversation_id,
            filename=upload_request.filename,
            content=upload_request.content,
            content_type=upload_request.content_type,
        )

        metadata = await document_service.get_metadata(
            upload_request.conversation_id,
            document_id,
        )

        task = asyncio.create_task(
            document_service.process_document(
                upload_request.conversation_id,
                document_id,
            )
        )
        _track_background_task(task)

        return DocumentUploadResponse(
            document_id=document_id,
            conversation_id=upload_request.conversation_id,
            filename=upload_request.filename,
            status=metadata.status.value,
            size_bytes=metadata.size_bytes,
            message="Document uploaded, processing started",
        )

    except Exception as e:
        logger.error(f"Document upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=_sanitize_error_message(e))


@router.get("/documents/{conversation_id}", response_model=DocumentListResponse)
async def list_documents(
    conversation_id: str,
    document_service: DocumentServiceDep,
) -> DocumentListResponse:
    """List all documents for a conversation."""
    if not document_service:
        raise HTTPException(status_code=503, detail="Document service not available")

    try:
        summaries = await document_service.get_summaries(conversation_id)

        documents = [
            DocumentSummaryResponse(
                document_id=s.document_id,
                filename=s.filename,
                status=s.status.value,
                overall_summary=s.overall_summary,
                key_topics=s.key_topics,
                document_type=s.document_type,
                relevance_hints=s.relevance_hints,
                chunk_count=s.chunk_count,
                total_tokens=s.total_tokens,
                processing_progress=s.processing_progress,
            )
            for s in summaries
        ]

        return DocumentListResponse(
            conversation_id=conversation_id,
            documents=documents,
            count=len(documents),
        )

    except Exception as e:
        logger.error(f"List documents failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_sanitize_error_message(e))


@router.get("/documents/{conversation_id}/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    conversation_id: str,
    document_id: str,
    document_service: DocumentServiceDep,
) -> DocumentStatusResponse:
    """Get processing status for a specific document."""
    if not document_service:
        raise HTTPException(status_code=503, detail="Document service not available")

    try:
        metadata = await document_service.get_metadata(conversation_id, document_id)

        return DocumentStatusResponse(
            document_id=document_id,
            filename=metadata.filename,
            status=metadata.status.value,
            processing_progress=metadata.processing_progress,
            error_message=metadata.error_message,
        )

    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        logger.error(f"Get document status failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_sanitize_error_message(e))


@router.delete("/documents/{conversation_id}/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    conversation_id: str,
    document_id: str,
    document_service: DocumentServiceDep,
) -> DocumentDeleteResponse:
    """Delete a document from a conversation."""
    if not document_service:
        raise HTTPException(status_code=503, detail="Document service not available")

    try:
        await document_service.delete_document(conversation_id, document_id)

        return DocumentDeleteResponse(
            success=True,
            document_id=document_id,
            message="Document deleted",
        )

    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        logger.error(f"Delete document failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_sanitize_error_message(e))
