"""FastAPI dependency injection."""

from typing import Annotated

from fastapi import Depends, Request

from agentic_chatbot.config.settings import Settings, get_settings
from agentic_chatbot.mcp.callbacks import ElicitationManager
from agentic_chatbot.mcp.manager import MCPClientManager
from agentic_chatbot.mcp.registry import MCPServerRegistry
from agentic_chatbot.mcp.session import MCPSessionManager
from agentic_chatbot.tools.provider import UnifiedToolProvider
from agentic_chatbot.tools.registry import LocalToolRegistry
from agentic_chatbot.operators.registry import OperatorRegistry


def get_app_settings() -> Settings:
    """Get application settings."""
    return get_settings()


async def get_mcp_registry(request: Request) -> MCPServerRegistry | None:
    """Get MCP server registry from application state."""
    return getattr(request.app.state, "mcp_server_registry", None)


async def get_mcp_client_manager(request: Request) -> MCPClientManager | None:
    """Get MCP client manager from application state."""
    return getattr(request.app.state, "mcp_client_manager", None)


async def get_mcp_session_manager(
    registry: Annotated[MCPServerRegistry | None, Depends(get_mcp_registry)],
    client_manager: Annotated[MCPClientManager | None, Depends(get_mcp_client_manager)],
) -> MCPSessionManager | None:
    """Get MCP session manager."""
    if registry and client_manager:
        return MCPSessionManager(client_manager, registry)
    return None


async def get_elicitation_manager(request: Request) -> ElicitationManager:
    """
    Get or create ElicitationManager from application state.

    The ElicitationManager is shared across requests to allow
    responses to be matched with pending elicitation requests.
    """
    if not hasattr(request.app.state, "elicitation_manager"):
        request.app.state.elicitation_manager = ElicitationManager()
    return request.app.state.elicitation_manager


async def get_tool_provider(
    mcp_registry: Annotated[MCPServerRegistry | None, Depends(get_mcp_registry)],
) -> UnifiedToolProvider:
    """
    Get UnifiedToolProvider for local and remote tools.

    The tool provider merges:
    - Local tools (zero-latency, in-process)
    - Remote MCP tools (network calls to external servers)

    This gives the supervisor a unified interface to all tools.
    """
    return UnifiedToolProvider(
        local_registry=LocalToolRegistry,
        mcp_registry=mcp_registry,
        operator_registry=OperatorRegistry,
    )


# Type aliases for dependency injection
SettingsDep = Annotated[Settings, Depends(get_app_settings)]
MCPRegistryDep = Annotated[MCPServerRegistry | None, Depends(get_mcp_registry)]
MCPClientManagerDep = Annotated[MCPClientManager | None, Depends(get_mcp_client_manager)]
MCPSessionManagerDep = Annotated[MCPSessionManager | None, Depends(get_mcp_session_manager)]
ElicitationManagerDep = Annotated[ElicitationManager, Depends(get_elicitation_manager)]
ToolProviderDep = Annotated[UnifiedToolProvider, Depends(get_tool_provider)]
