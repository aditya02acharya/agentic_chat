"""AWS Bedrock provider for Claude models."""

import json
from typing import AsyncIterator

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agentic_chatbot.utils.providers.base import BaseLLMProvider, LLMResponse
from agentic_chatbot.utils.logging import get_logger


logger = get_logger(__name__)


class BedrockProvider(BaseLLMProvider):
    """
    AWS Bedrock provider for Claude models.

    Uses boto3 bedrock-runtime client to call Claude models via AWS Bedrock.

    Features:
    - Model aliases mapped to Bedrock model IDs
    - Cross-region inference support
    - Automatic retry with exponential backoff
    - Usage tracking (tokens)
    - Streaming support

    Note: Requires AWS credentials configured via:
    - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    - AWS credentials file (~/.aws/credentials)
    - IAM role (when running on AWS)
    """

    # Bedrock uses different model IDs - includes anthropic. prefix
    MODEL_ALIASES: dict[str, str] = {
        "haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "opus": "anthropic.claude-3-opus-20240229-v1:0",
        # Also support cross-region inference IDs
        "haiku-us": "us.anthropic.claude-3-haiku-20240307-v1:0",
        "sonnet-us": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "opus-us": "us.anthropic.claude-3-opus-20240229-v1:0",
    }

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    )
    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str = "sonnet",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Call Bedrock API and get complete response."""
        resolved_model = self.resolve_model(model)

        logger.debug(
            "Calling Bedrock API",
            model=resolved_model,
            region=self._region_name,
            prompt_length=len(prompt),
            system_length=len(system),
        )

        # Build Anthropic messages format (Bedrock uses same format)
        messages = [{"role": "user", "content": prompt}]

        # Build request body
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        if system:
            body["system"] = system

        session = self._create_session()
        async with session.client(
            "bedrock-runtime",
            region_name=self._region_name,
        ) as client:
            response = await client.invoke_model(
                modelId=resolved_model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )

            # Read response body
            response_body = await response["body"].read()
            result = json.loads(response_body)

        # Extract content from response
        content = ""
        if result.get("content") and len(result["content"]) > 0:
            content = result["content"][0].get("text", "")

        # Extract usage info
        usage = result.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        stop_reason = result.get("stop_reason", "unknown")

        logger.debug(
            "Bedrock response received",
            model=resolved_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
        )

        return LLMResponse(
            content=content,
            model=resolved_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
            provider=self.provider_name,
        )

    async def stream(
        self,
        prompt: str,
        system: str = "",
        model: str = "sonnet",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Stream response from Bedrock API."""
        resolved_model = self.resolve_model(model)

        logger.debug(
            "Starting Bedrock stream",
            model=resolved_model,
            region=self._region_name,
            prompt_length=len(prompt),
        )

        # Build Anthropic messages format
        messages = [{"role": "user", "content": prompt}]

        # Build request body
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        if system:
            body["system"] = system

        session = self._create_session()
        async with session.client(
            "bedrock-runtime",
            region_name=self._region_name,
        ) as client:
            response = await client.invoke_model_with_response_stream(
                modelId=resolved_model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )

            # Process event stream
            async for event in response["body"]:
                chunk = event.get("chunk")
                if chunk:
                    chunk_data = json.loads(chunk["bytes"])

                    # Handle different event types
                    event_type = chunk_data.get("type")

                    if event_type == "content_block_delta":
                        delta = chunk_data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                yield text

                    elif event_type == "message_delta":
                        # End of message, could extract stop_reason here
                        pass
