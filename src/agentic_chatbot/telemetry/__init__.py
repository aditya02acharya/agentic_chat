"""Telemetry and observability module.

Provides OpenTelemetry integration for:
- LLM call tracing (detailed token usage)
- Performance monitoring
- Cost tracking
- Audit logs

This module uses placeholders until OpenTelemetry SDK is installed.
"""

from agentic_chatbot.telemetry.tracing import (
    get_tracer,
    trace_llm_call,
    LLMCallAttributes,
)

__all__ = [
    "get_tracer",
    "trace_llm_call",
    "LLMCallAttributes",
]
