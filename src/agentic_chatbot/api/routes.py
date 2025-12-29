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
    TokenUsageResponse,
    ErrorResponse,
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
    CognitionServiceDep,
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
    tool_provider: ToolProviderDep,
) -> ToolsResponse:
    """
    List available tools.

    Returns summaries of all registered tools:
    - Local tools (self-awareness, introspection)
    - Remote MCP tools (external servers)
    - Operators (internal execution strategies)
    """
    tools = []

    # Get all tool summaries via UnifiedToolProvider
    # This includes local tools, MCP tools, and can be extended
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
        logger.warning(f"Failed to get tool summaries from provider: {e}")

    # Get operator summaries (operators are separate from tools)
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
    elicitation_manager: ElicitationManagerDep,
    tool_provider: ToolProviderDep,
    cognition_service: CognitionServiceDep,
) -> StreamingResponse:
    """
    Main chat endpoint with SSE streaming.

    Returns Server-Sent Events stream with progress updates
    and response chunks.

    Events include:
    - supervisor.thinking: Agent is analyzing the query
    - supervisor.decided: Agent made a decision
    - tool.start/progress/complete/error: Tool execution status
    - mcp.progress/content/elicitation/error: MCP-specific events
    - workflow.*: Workflow execution events
    - response.chunk/done: Response streaming
    - clarify.request: Clarification needed from user
    - error: General errors

    For elicitation events (mcp.elicitation), the UI should:
    1. Display the prompt to the user
    2. Collect user input
    3. Submit response via POST /elicitation/respond
    """
    # Check if shutdown is in progress
    if getattr(request.app.state, "is_shutting_down", False):
        raise HTTPException(status_code=503, detail="Service is shutting down")

    # Create request context
    event_queue: asyncio.Queue[Event] = asyncio.Queue()
    request_id = f"{chat_request.conversation_id}_{asyncio.get_event_loop().time()}"

    # Create event emitter for this request
    emitter = EventEmitter(event_queue)

    # Create MCP callbacks wired to the event emitter
    mcp_callbacks = create_mcp_callbacks(
        emitter=emitter,
        elicitation_manager=elicitation_manager,
        request_id=request_id,
    )

    # Create initial state for LangGraph
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
        cognition_service=cognition_service,
    )

    # Track request
    active_requests = getattr(request.app.state, "active_requests", set())
    active_requests.add(request_id)

    async def run_graph() -> None:
        """Run the LangGraph in background with timeout protection."""
        try:
            # Create the chat graph
            graph = create_chat_graph()

            # Configuration for LangGraph
            config = {
                "configurable": {
                    "thread_id": chat_request.conversation_id,
                }
            }

            # Run the graph with timeout protection
            await asyncio.wait_for(
                graph.ainvoke(initial_state, config),
                timeout=GRAPH_EXECUTION_TIMEOUT,
            )

        except asyncio.TimeoutError:
            logger.warning(
                f"Graph execution timeout after {GRAPH_EXECUTION_TIMEOUT}s",
                request_id=request_id,
            )
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
            # Ensure done event is sent
            await event_queue.put(
                ResponseDoneEvent.create(request_id=request_id)
            )
            # Untrack request
            active_requests.discard(request_id)

    # Start graph execution
    task = asyncio.create_task(run_graph())

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
    cognition_service: CognitionServiceDep,
) -> ChatResponse:
    """
    Non-streaming chat endpoint.

    Alternative to SSE streaming for simpler integrations.
    Waits for complete response before returning.

    Note: Elicitation requests cannot be handled in sync mode
    as they require interactive user input.
    """
    # Check if shutdown is in progress
    if getattr(request.app.state, "is_shutting_down", False):
        raise HTTPException(status_code=503, detail="Service is shutting down")

    request_id = f"{chat_request.conversation_id}_{asyncio.get_event_loop().time()}"

    # Create event emitter (events go to queue but aren't streamed)
    event_queue: asyncio.Queue[Event] = asyncio.Queue()
    emitter = EventEmitter(event_queue)

    # Create MCP callbacks
    mcp_callbacks = create_mcp_callbacks(
        emitter=emitter,
        elicitation_manager=elicitation_manager,
        request_id=request_id,
    )

    # Create initial state
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
        cognition_service=cognition_service,
    )

    try:
        # Create and run graph with timeout protection
        graph = create_chat_graph()
        config = {
            "configurable": {
                "thread_id": chat_request.conversation_id,
            }
        }

        try:
            final_state = await asyncio.wait_for(
                graph.ainvoke(initial_state, config),
                timeout=GRAPH_EXECUTION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Sync graph execution timeout after {GRAPH_EXECUTION_TIMEOUT}s",
                request_id=request_id,
            )
            raise HTTPException(
                status_code=504,
                detail=f"Request timed out after {GRAPH_EXECUTION_TIMEOUT} seconds",
            )

        # Get response from final state
        response = final_state.get("final_response", "")

        # Extract token usage from final state
        token_usage = final_state.get("token_usage")
        usage_response = None
        if token_usage:
            usage_response = TokenUsageResponse(
                input_tokens=token_usage.input_tokens,
                output_tokens=token_usage.output_tokens,
                thinking_tokens=token_usage.thinking_tokens,
                cache_read_tokens=token_usage.cache_read_tokens,
                cache_write_tokens=token_usage.cache_write_tokens,
                total_tokens=token_usage.total,
            )

        return ChatResponse(
            conversation_id=chat_request.conversation_id,
            response=response,
            request_id=request_id,
            usage=usage_response,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ELICITATION ENDPOINTS
# =============================================================================


@router.post("/elicitation/respond", response_model=ElicitationResponseResult)
async def submit_elicitation_response(
    elicitation_request: ElicitationResponseRequest,
    elicitation_manager: ElicitationManagerDep,
) -> ElicitationResponseResult:
    """
    Submit user response to a pending elicitation request.

    When an MCP tool needs user input, it emits an mcp.elicitation event
    with an elicitation_id. The UI should:
    1. Display the prompt to the user
    2. Collect the user's input
    3. Call this endpoint with the elicitation_id and value

    This unblocks the waiting tool and allows it to continue execution.

    Args:
        elicitation_id: ID from the mcp.elicitation event
        value: User's response (string, choice value, or boolean for confirm)
        cancelled: Set to true if user cancelled/dismissed the prompt
    """
    success = await elicitation_manager.submit_response(
        elicitation_id=elicitation_request.elicitation_id,
        value=elicitation_request.value,
        cancelled=elicitation_request.cancelled,
    )

    if success:
        return ElicitationResponseResult(
            success=True,
            elicitation_id=elicitation_request.elicitation_id,
            message="Response accepted",
        )
    else:
        return ElicitationResponseResult(
            success=False,
            elicitation_id=elicitation_request.elicitation_id,
            message="Elicitation not found or already responded",
        )


@router.get("/elicitation/pending", response_model=PendingElicitationsResponse)
async def list_pending_elicitations(
    elicitation_manager: ElicitationManagerDep,
) -> PendingElicitationsResponse:
    """
    List all pending elicitation requests.

    Useful for:
    - Debugging to see what's waiting for user input
    - Reconnecting after page refresh to see pending prompts
    - Admin dashboards monitoring tool activity
    """
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
    return PendingElicitationsResponse(
        elicitations=elicitations,
        count=len(elicitations),
    )


@router.delete("/elicitation/{elicitation_id}")
async def cancel_elicitation(
    elicitation_id: str,
    elicitation_manager: ElicitationManagerDep,
) -> ElicitationResponseResult:
    """
    Cancel a pending elicitation request.

    The tool will receive a cancelled response and should
    handle it gracefully (e.g., use a default value or skip
    the operation).
    """
    success = await elicitation_manager.cancel_elicitation(elicitation_id)

    if success:
        return ElicitationResponseResult(
            success=True,
            elicitation_id=elicitation_id,
            message="Elicitation cancelled",
        )
    else:
        return ElicitationResponseResult(
            success=False,
            elicitation_id=elicitation_id,
            message="Elicitation not found or already responded",
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
    """
    Upload a document for conversation context.

    Documents are processed asynchronously:
    1. Document is saved and chunked
    2. Each chunk is summarized by LLM
    3. Overall summary is generated

    The supervisor can use summaries to decide which documents
    to load into context for answering questions.

    Limits:
    - Maximum 5 documents per conversation
    - Maximum 1MB per document
    - Text files only (text/plain, text/markdown)
    """
    if not document_service:
        raise HTTPException(
            status_code=503,
            detail="Document service not available",
        )

    try:
        # Create document
        document_id = await document_service.create_document(
            conversation_id=upload_request.conversation_id,
            filename=upload_request.filename,
            content=upload_request.content,
            content_type=upload_request.content_type,
        )

        # Get metadata for response
        metadata = await document_service.get_metadata(
            upload_request.conversation_id,
            document_id,
        )

        # Start background processing
        asyncio.create_task(
            document_service.process_document(
                upload_request.conversation_id,
                document_id,
            )
        )

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
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/documents/{conversation_id}",
    response_model=DocumentListResponse,
)
async def list_documents(
    conversation_id: str,
    document_service: DocumentServiceDep,
) -> DocumentListResponse:
    """
    List all documents for a conversation.

    Returns summaries and processing status for each document.
    """
    if not document_service:
        raise HTTPException(
            status_code=503,
            detail="Document service not available",
        )

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
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/documents/{conversation_id}/{document_id}/status",
    response_model=DocumentStatusResponse,
)
async def get_document_status(
    conversation_id: str,
    document_id: str,
    document_service: DocumentServiceDep,
) -> DocumentStatusResponse:
    """
    Get processing status for a specific document.

    Useful for polling during processing.
    """
    if not document_service:
        raise HTTPException(
            status_code=503,
            detail="Document service not available",
        )

    try:
        metadata = await document_service.get_metadata(
            conversation_id,
            document_id,
        )

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
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/documents/{conversation_id}/{document_id}",
    response_model=DocumentDeleteResponse,
)
async def delete_document(
    conversation_id: str,
    document_id: str,
    document_service: DocumentServiceDep,
) -> DocumentDeleteResponse:
    """
    Delete a document from a conversation.
    """
    if not document_service:
        raise HTTPException(
            status_code=503,
            detail="Document service not available",
        )

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
        raise HTTPException(status_code=500, detail=str(e))
