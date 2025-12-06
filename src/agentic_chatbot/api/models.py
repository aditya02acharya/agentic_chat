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
    context: dict[str, Any] | None = Field(None, description="Additional context")


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


class ChatResponse(BaseModel):
    """
    Non-streaming response for chat endpoint.

    Note: Primary chat interface uses SSE streaming.
    This is for non-streaming fallback.
    """

    conversation_id: str = Field(..., description="Conversation identifier")
    response: str = Field(..., description="Assistant's response")
    request_id: str = Field(..., description="Request identifier")


class ErrorResponse(BaseModel):
    """Error response body."""

    error: str = Field(..., description="Error message")
    error_type: str = Field("general", description="Error type")
    request_id: str | None = Field(None, description="Request identifier if available")
