"""Pydantic request/response schemas for API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# REQUEST MODELS
# =============================================================================


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    conversation_id: str = Field(..., description="Unique conversation identifier")
    message: str = Field(..., description="User's message")
    user_id: str | None = Field(None, description="User identifier for personalization and memory")
    context: dict[str, Any] | None = Field(None, description="Additional context")
    model: str | None = Field(
        None,
        description="Model to use for response generation (e.g., 'sonnet', 'haiku', 'thinking'). Uses default if not specified.",
    )


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class HealthResponse(BaseModel):
    """Response for health check endpoint."""

    status: str = Field("ok", description="Health status")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ToolSummaryResponse(BaseModel):
    """Summary of a single tool."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    server_id: str | None = Field(None, description="MCP server hosting this tool")


class ToolsResponse(BaseModel):
    """Response for tools listing endpoint."""

    tools: list[ToolSummaryResponse] = Field(default_factory=list)
    count: int = Field(0, description="Total number of tools")


class MessageResponse(BaseModel):
    """A single message in conversation history."""

    role: str = Field(..., description="Message role (user, assistant)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatHistoryResponse(BaseModel):
    """Response for chat history endpoint."""

    conversation_id: str = Field(..., description="Conversation identifier")
    messages: list[MessageResponse] = Field(default_factory=list)


class TokenUsageResponse(BaseModel):
    """Token usage information."""

    input_tokens: int = Field(0, description="Number of input tokens")
    output_tokens: int = Field(0, description="Number of output tokens")
    thinking_tokens: int = Field(0, description="Number of thinking tokens (extended thinking)")
    cache_read_tokens: int = Field(0, description="Number of cached tokens read")
    cache_write_tokens: int = Field(0, description="Number of tokens written to cache")
    total_tokens: int = Field(0, description="Total tokens used")


class ChatResponse(BaseModel):
    """
    Non-streaming response for chat endpoint.

    Note: Primary chat interface uses SSE streaming.
    This is for non-streaming fallback.
    """

    conversation_id: str = Field(..., description="Conversation identifier")
    response: str = Field(..., description="Assistant's response")
    request_id: str = Field(..., description="Request identifier")
    usage: TokenUsageResponse | None = Field(None, description="Token usage information")


class ErrorResponse(BaseModel):
    """Error response body."""

    error: str = Field(..., description="Error message")
    error_type: str = Field("general", description="Error type")
    request_id: str | None = Field(None, description="Request identifier if available")


# =============================================================================
# ELICITATION MODELS
# =============================================================================


class ElicitationResponseRequest(BaseModel):
    """Request body for submitting user response to elicitation."""

    elicitation_id: str = Field(..., description="ID of the elicitation request")
    value: Any = Field(..., description="User's response value")
    cancelled: bool = Field(False, description="Whether user cancelled the request")


class ElicitationResponseResult(BaseModel):
    """Response for elicitation submission."""

    success: bool = Field(..., description="Whether response was accepted")
    elicitation_id: str = Field(..., description="ID of the elicitation request")
    message: str = Field("", description="Additional information")


class PendingElicitationResponse(BaseModel):
    """Information about a pending elicitation."""

    elicitation_id: str = Field(..., description="Unique elicitation identifier")
    server_id: str = Field(..., description="MCP server ID")
    tool_name: str = Field(..., description="Tool requesting input")
    prompt: str = Field(..., description="Question for the user")
    input_type: str = Field("text", description="Expected input type")
    options: list[str] | None = Field(None, description="Options for choice input")
    default: str | None = Field(None, description="Default value")
    timeout_seconds: float = Field(60.0, description="Timeout for response")


class PendingElicitationsResponse(BaseModel):
    """Response listing all pending elicitations."""

    elicitations: list[PendingElicitationResponse] = Field(default_factory=list)
    count: int = Field(0, description="Number of pending elicitations")


# =============================================================================
# DOCUMENT MODELS
# =============================================================================


class DocumentUploadRequest(BaseModel):
    """Request body for document upload."""

    conversation_id: str = Field(..., description="Conversation to attach document to")
    filename: str = Field(..., description="Original filename")
    content: str = Field(..., description="Document text content")
    content_type: str = Field(
        "text/plain",
        description="MIME type (text/plain or text/markdown)",
    )


class DocumentUploadResponse(BaseModel):
    """Response for document upload."""

    document_id: str = Field(..., description="Unique document identifier")
    conversation_id: str = Field(..., description="Conversation ID")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Processing status")
    size_bytes: int = Field(..., description="Document size in bytes")
    message: str = Field("", description="Additional information")


class DocumentStatusResponse(BaseModel):
    """Document processing status."""

    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Processing status")
    processing_progress: float = Field(0.0, description="Progress 0.0-1.0")
    error_message: str | None = Field(None, description="Error if failed")


class DocumentSummaryResponse(BaseModel):
    """Document summary for API response."""

    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Processing status")
    overall_summary: str = Field("", description="Document summary")
    key_topics: list[str] = Field(default_factory=list, description="Key topics")
    document_type: str = Field("unknown", description="Document type")
    relevance_hints: str = Field("", description="When to use this document")
    chunk_count: int = Field(0, description="Number of chunks")
    total_tokens: int = Field(0, description="Estimated token count")
    processing_progress: float = Field(0.0, description="Progress 0.0-1.0")


class DocumentListResponse(BaseModel):
    """Response listing all documents for a conversation."""

    conversation_id: str = Field(..., description="Conversation identifier")
    documents: list[DocumentSummaryResponse] = Field(default_factory=list)
    count: int = Field(0, description="Number of documents")


class DocumentDeleteResponse(BaseModel):
    """Response for document deletion."""

    success: bool = Field(..., description="Whether deletion succeeded")
    document_id: str = Field(..., description="Deleted document ID")
    message: str = Field("", description="Additional information")
