"""OpenTelemetry tracing for LLM calls.

This module provides Level 2 token tracking via OTEL spans.
Each LLM call creates a span with detailed attributes:
- Model used
- Operation type (supervisor, writer, tool_call, etc.)
- Input/output token counts
- Latency
- Cost estimates

PLACEHOLDER IMPLEMENTATION:
Currently uses noop tracer. To enable OTEL:
1. Install: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
2. Set OTEL_ENABLED=true in environment
3. Configure OTEL_EXPORTER_OTLP_ENDPOINT for your collector

For production:
- Export to Jaeger, Zipkin, or cloud providers (Datadog, New Relic, etc.)
- Use batch span processor for performance
- Add resource attributes (service name, version, environment)
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from agentic_chatbot.config.settings import get_settings
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


# =============================================================================
# OTEL PLACEHOLDER (NoOp Tracer)
# =============================================================================
# Replace with real OpenTelemetry SDK when installed


class NoOpSpan:
    """Placeholder span that does nothing."""

    def set_attribute(self, key: str, value: Any) -> None:
        """Record an attribute (noop)."""
        pass

    def set_status(self, status: str, description: str = "") -> None:
        """Set span status (noop)."""
        pass

    def record_exception(self, exception: Exception) -> None:
        """Record an exception (noop)."""
        pass

    def end(self) -> None:
        """End the span (noop)."""
        pass


class NoOpTracer:
    """Placeholder tracer that does nothing."""

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[NoOpSpan]:
        """Start a new span (noop)."""
        span = NoOpSpan()
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        try:
            yield span
        finally:
            span.end()


# Global noop tracer instance
_tracer = NoOpTracer()


def get_tracer() -> NoOpTracer:
    """
    Get the OTEL tracer for LLM calls.

    Returns NoOp tracer until OTEL is enabled.

    To enable real tracing:
    1. Install opentelemetry-api and opentelemetry-sdk
    2. Set OTEL_ENABLED=true
    3. Replace this with:
        from opentelemetry import trace
        return trace.get_tracer("agentic_chatbot.llm")
    """
    settings = get_settings()

    # Placeholder: Return noop tracer
    # TODO: When OTEL_ENABLED=true, return real tracer:
    # if settings.otel_enabled:
    #     from opentelemetry import trace
    #     return trace.get_tracer("agentic_chatbot.llm", version=__version__)

    return _tracer


# =============================================================================
# LLM CALL ATTRIBUTES (Level 2 Tracking)
# =============================================================================


@dataclass
class LLMCallAttributes:
    """
    Attributes for an LLM call span.

    These are recorded as span attributes in OTEL for detailed analysis.
    """

    # Operation context
    operation: str  # e.g., "supervisor", "writer", "synthesizer", "tool_call"
    conversation_id: str
    request_id: str

    # Model details
    model: str
    provider: str  # "anthropic", "bedrock"

    # Token metrics (Level 2: Per-call breakdown)
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    # Performance
    latency_ms: float = 0.0

    # Additional context
    thinking_enabled: bool = False
    streaming: bool = False
    temperature: float = 1.0

    def to_span_attributes(self) -> dict[str, Any]:
        """Convert to OTEL span attributes."""
        return {
            # Operation
            "llm.operation": self.operation,
            "llm.conversation_id": self.conversation_id,
            "llm.request_id": self.request_id,
            # Model
            "llm.model": self.model,
            "llm.provider": self.provider,
            # Tokens (OpenTelemetry semantic conventions for LLM)
            "llm.usage.input_tokens": self.input_tokens,
            "llm.usage.output_tokens": self.output_tokens,
            "llm.usage.thinking_tokens": self.thinking_tokens,
            "llm.usage.cache_read_tokens": self.cache_read_tokens,
            "llm.usage.cache_write_tokens": self.cache_write_tokens,
            "llm.usage.total_tokens": (
                self.input_tokens
                + self.output_tokens
                + self.thinking_tokens
                + self.cache_read_tokens
                + self.cache_write_tokens
            ),
            # Performance
            "llm.latency_ms": self.latency_ms,
            # Config
            "llm.thinking_enabled": self.thinking_enabled,
            "llm.streaming": self.streaming,
            "llm.temperature": self.temperature,
        }


@contextmanager
def trace_llm_call(
    operation: str,
    model: str,
    conversation_id: str = "",
    request_id: str = "",
    provider: str = "anthropic",
) -> Iterator[LLMCallAttributes]:
    """
    Trace an LLM call with OTEL span.

    Usage:
        with trace_llm_call("supervisor", model="claude-3-5-sonnet", ...) as attrs:
            response = await llm.complete(...)
            # Populate attributes from response
            attrs.input_tokens = response.usage.input_tokens
            attrs.output_tokens = response.usage.output_tokens
            attrs.latency_ms = response.latency_ms

    The span will automatically record all attributes when exiting the context.

    Args:
        operation: Operation type (supervisor, writer, synthesizer, etc.)
        model: Model identifier
        conversation_id: Conversation ID
        request_id: Request ID
        provider: LLM provider

    Yields:
        LLMCallAttributes object to populate during the call
    """
    tracer = get_tracer()
    attrs = LLMCallAttributes(
        operation=operation,
        model=model,
        conversation_id=conversation_id,
        request_id=request_id,
        provider=provider,
    )

    start_time = time.perf_counter()

    with tracer.start_as_current_span(
        f"llm.{operation}",
        attributes={
            "llm.operation": operation,
            "llm.model": model,
        },
    ) as span:
        try:
            yield attrs

            # Calculate latency
            attrs.latency_ms = (time.perf_counter() - start_time) * 1000

            # Record all attributes
            for key, value in attrs.to_span_attributes().items():
                span.set_attribute(key, value)

            # Mark as successful
            span.set_status("OK")

        except Exception as e:
            # Record exception
            span.record_exception(e)
            span.set_status("ERROR", description=str(e))
            raise


# =============================================================================
# FUTURE: OTEL INITIALIZATION
# =============================================================================
# When OTEL is enabled, initialize with:
#
# from opentelemetry import trace
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import BatchSpanProcessor
# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
# from opentelemetry.sdk.resources import Resource
#
# def initialize_otel():
#     """Initialize OpenTelemetry tracing."""
#     resource = Resource.create({
#         "service.name": "agentic-chatbot",
#         "service.version": __version__,
#         "deployment.environment": get_settings().environment,
#     })
#
#     provider = TracerProvider(resource=resource)
#     processor = BatchSpanProcessor(OTLPSpanExporter())
#     provider.add_span_processor(processor)
#     trace.set_tracer_provider(provider)
