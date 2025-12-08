"""FastAPI dependency injection."""

from typing import TYPE_CHECKING

from ..config.settings import get_settings, Settings

if TYPE_CHECKING:
    from ..app import Application


_app_instance: "Application | None" = None


def set_app_instance(app: "Application") -> None:
    """Set the global application instance."""
    global _app_instance
    _app_instance = app


def get_app() -> "Application":
    """Get the application instance."""
    if _app_instance is None:
        raise RuntimeError("Application not initialized")
    return _app_instance


def get_settings_dep() -> Settings:
    """Dependency for getting settings."""
    return get_settings()
