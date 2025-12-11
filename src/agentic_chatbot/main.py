"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentic_chatbot import __version__
from agentic_chatbot.api.routes import router
from agentic_chatbot.app import app_instance, setup_signal_handlers
from agentic_chatbot.config.settings import get_settings
from agentic_chatbot.utils.logging import configure_logging, get_logger

# Import operators to trigger registration
from agentic_chatbot.operators.llm import (  # noqa: F401
    QueryRewriterOperator,
    SynthesizerOperator,
    WriterOperator,
    AnalyzerOperator,
)
from agentic_chatbot.operators.mcp import (  # noqa: F401
    RAGRetrieverOperator,
    WebSearcherOperator,
)
from agentic_chatbot.operators.hybrid import CoderOperator  # noqa: F401

# Register builtin local tools (self-awareness, introspection)
from agentic_chatbot.tools.builtin import register_builtin_tools
register_builtin_tools()


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    await app_instance.startup()

    # Store references in app state
    app.state.mcp_server_registry = app_instance.mcp_server_registry
    app.state.mcp_client_manager = app_instance.mcp_client_manager
    app.state.active_requests = app_instance.active_requests
    app.state.is_shutting_down = False

    # Setup signal handlers
    setup_signal_handlers(app_instance)

    yield

    # Shutdown
    app.state.is_shutting_down = True
    await app_instance.shutdown()


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI application
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Agentic Chatbot API",
        description="ReACT-based agentic chatbot with MCP integration",
        version=__version__,
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(router, prefix="/api/v1")

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": "Agentic Chatbot API",
            "version": __version__,
            "docs": "/docs",
        }

    return app


# Create application instance
app = create_app()


def main() -> None:
    """Entry point for running the application."""
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "agentic_chatbot.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
