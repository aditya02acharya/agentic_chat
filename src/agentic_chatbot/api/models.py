"""API request/response models."""

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    conversation_id: str = Field(description="Unique conversation identifier")
    message: str = Field(description="User message")
    context: dict[str, Any] | None = Field(
        default=None, description="Optional additional context"
    )


class ChatResponse(BaseModel):
    """Response model for non-streaming chat."""

    conversation_id: str
    response: str
    request_id: str


class ToolInfo(BaseModel):
    """Information about an available tool."""

    name: str
    description: str
    type: str


class ToolsResponse(BaseModel):
    """Response model for tools endpoint."""

    tools: list[ToolInfo]


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str
