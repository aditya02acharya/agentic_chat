"""AWS Bedrock provider for Claude models using Converse API.

Resilience patterns applied:
- Retry with exponential backoff for transient failures
- Circuit breaker to prevent cascade failures to overloaded API
- Timeout to bound operation duration
"""

from typing import AsyncIterator, Any

from agentic_chatbot.config.models import ThinkingConfig, TokenUsage
from agentic_chatbot.core.resilience import (
    llm_retry,
    llm_circuit_breaker,
    llm_timeout,
    wrap_aws_errors,
    TransientError,
    RateLimitError,
    BreakerOpen,
)
from agentic_chatbot.utils.providers.base import BaseLLMProvider, LLMResponse
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class BedrockProvider(BaseLLMProvider):
    """
    AWS Bedrock provider for Claude models using the Converse API.

    Uses the Bedrock Converse API which provides a unified interface
    for conversational AI models across different providers.

    Features:
    - Model resolution via ModelRegistry
    - Cross-region inference support
    - Automatic retry with exponential backoff
    - Comprehensive token tracking
    - Streaming support via converse_stream

    Note: Requires AWS credentials configured via:
    - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    - AWS credentials file (~/.aws/credentials)
    - IAM role (when running on AWS)

    Note: Extended thinking is not yet supported on Bedrock.
    """

    def __init__(
        self,
        region_name: str = "us-east-1",
        profile_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ):
        """
        Initialize Bedrock provider.

        Args:
            region_name: AWS region for Bedrock (default: us-east-1)
            profile_name: AWS profile name (optional)
            aws_access_key_id: AWS access key ID (optional, uses default chain if not provided)
            aws_secret_access_key: AWS secret access key (optional)
        """
        try:
            import aioboto3
        except ImportError:
            raise ImportError(
                "aioboto3 is required for Bedrock provider. "
                "Install it with: pip install aioboto3"
            )

        self._region_name = region_name
        self._session_kwargs: dict = {}

        if profile_name:
            self._session_kwargs["profile_name"] = profile_name
        if aws_access_key_id and aws_secret_access_key:
            self._session_kwargs["aws_access_key_id"] = aws_access_key_id
            self._session_kwargs["aws_secret_access_key"] = aws_secret_access_key

        self._aioboto3 = aioboto3

    @property
    def provider_name(self) -> str:
        return "bedrock"

    def _create_session(self):
        """Create a new aioboto3 session."""
        return self._aioboto3.Session(**self._session_kwargs)

    def _build_messages(self, prompt: str) -> list[dict]:
        """Build messages in Converse API format."""
        return [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ]

    def _build_system(self, system: str) -> list[dict] | None:
        """Build system prompt in Converse API format."""
        if not system:
            return None
        return [{"text": system}]

    @llm_retry
    @llm_circuit_breaker
    @llm_timeout
    @wrap_aws_errors
    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str = "sonnet",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        thinking: ThinkingConfig | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Call Bedrock Converse API and get complete response.

        Note: Extended thinking is not yet supported on Bedrock.
        The thinking parameter is accepted for API compatibility but ignored.
        """
        resolved_model = self.resolve_model(model)

        if thinking and thinking.enabled:
            logger.warning(
                "Extended thinking is not supported on Bedrock, ignoring",
                model=resolved_model,
            )

        logger.debug(
            "Calling Bedrock Converse API",
            model=resolved_model,
            region=self._region_name,
            prompt_length=len(prompt),
            system_length=len(system),
        )

        # Build request parameters
        messages = self._build_messages(prompt)
        inference_config = {
            "maxTokens": max_tokens,
            "temperature": temperature,
        }

        # Build request kwargs
        request_kwargs = {
            "modelId": resolved_model,
            "messages": messages,
            "inferenceConfig": inference_config,
        }

        # Add system prompt if provided
        system_prompt = self._build_system(system)
        if system_prompt:
            request_kwargs["system"] = system_prompt

        session = self._create_session()
        async with session.client(
            "bedrock-runtime",
            region_name=self._region_name,
        ) as client:
            response = await client.converse(**request_kwargs)

        # Extract content from response
        content = ""
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        for block in content_blocks:
            if "text" in block:
                content += block["text"]

        # Extract usage info
        usage_data = response.get("usage", {})
        usage = TokenUsage(
            input_tokens=usage_data.get("inputTokens", 0),
            output_tokens=usage_data.get("outputTokens", 0),
        )
        stop_reason = response.get("stopReason", "unknown")

        logger.debug(
            "Bedrock Converse response received",
            model=resolved_model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            stop_reason=stop_reason,
        )

        return LLMResponse(
            content=content,
            model=resolved_model,
            usage=usage,
            stop_reason=stop_reason,
            provider=self.provider_name,
        )

    @llm_retry
    @llm_circuit_breaker
    @wrap_aws_errors
    async def stream(
        self,
        prompt: str,
        system: str = "",
        model: str = "sonnet",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        thinking: ThinkingConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream response from Bedrock Converse API.

        Resilience: Retry with exponential backoff + circuit breaker.
        Note: No timeout on stream as responses can be legitimately long.
        """
        resolved_model = self.resolve_model(model)

        if thinking and thinking.enabled:
            logger.warning(
                "Extended thinking is not supported on Bedrock streaming, ignoring",
                model=resolved_model,
            )

        logger.debug(
            "Starting Bedrock Converse stream",
            model=resolved_model,
            region=self._region_name,
            prompt_length=len(prompt),
        )

        # Build request parameters
        messages = self._build_messages(prompt)
        inference_config = {
            "maxTokens": max_tokens,
            "temperature": temperature,
        }

        # Build request kwargs
        request_kwargs = {
            "modelId": resolved_model,
            "messages": messages,
            "inferenceConfig": inference_config,
        }

        # Add system prompt if provided
        system_prompt = self._build_system(system)
        if system_prompt:
            request_kwargs["system"] = system_prompt

        session = self._create_session()
        async with session.client(
            "bedrock-runtime",
            region_name=self._region_name,
        ) as client:
            response = await client.converse_stream(**request_kwargs)

            # Process event stream
            async for event in response["stream"]:
                # Handle content block delta events
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        text = delta["text"]
                        if text:
                            yield text

                # messageStop and metadata events can be used for
                # stop_reason and usage tracking if needed
                elif "messageStop" in event:
                    # Could extract stopReason here
                    pass
                elif "metadata" in event:
                    # Could extract usage here
                    pass
