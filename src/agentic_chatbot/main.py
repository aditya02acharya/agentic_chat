"""FastAPI application entry point."""

import asyncio
import signal
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .api.dependencies import set_app_instance
from .app import Application
from .config.settings import get_settings
from .utils.logging import setup_logging, get_logger

application = Application()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()
    setup_logging(settings.log_level)

    set_app_instance(application)
    await application.startup()

    yield

    await application.shutdown()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Agentic Chatbot Backend",
        description="A ReACT-based supervisor with MCP integration",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app


app = create_app()


def main():
    """Run the application."""
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
