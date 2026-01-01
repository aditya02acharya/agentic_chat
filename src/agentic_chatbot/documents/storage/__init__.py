"""Document storage abstraction layer."""

from agentic_chatbot.documents.storage.base import DocumentStorage
from agentic_chatbot.documents.storage.local import LocalDocumentStorage

__all__ = [
    "DocumentStorage",
    "LocalDocumentStorage",
]
