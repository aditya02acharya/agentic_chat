# Developer Guide: Building an Agentic Chatbot from Scratch

A step-by-step tutorial for building a ReACT-based agentic chatbot using LangGraph. Follow these 10 phases in order to understand the architecture and build the system incrementally.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Phase 1: Foundation Layer](#phase-1-foundation-layer)
4. [Phase 2: LLM Integration](#phase-2-llm-integration)
5. [Phase 3: Events & Messaging](#phase-3-events--messaging)
6. [Phase 4: MCP Integration](#phase-4-mcp-integration)
7. [Phase 5: Document Management](#phase-5-document-management)
8. [Phase 6: Operators & Tools](#phase-6-operators--tools)
9. [Phase 7: Context Management](#phase-7-context-management)
10. [Phase 8: Graph & Nodes](#phase-8-graph--nodes)
11. [Phase 9: Cognition (Optional)](#phase-9-cognition-optional)
12. [Phase 10: API Layer & Application](#phase-10-api-layer--application)
13. [Running the Application](#running-the-application)
14. [Next Steps](#next-steps)

---

## Overview

This guide walks you through building an agentic chatbot that can:
- Answer questions directly using an LLM
- Call tools and operators to retrieve external data
- Create multi-step workflows for complex tasks
- Ask for clarification when needed
- Stream responses in real-time via SSE
- Remember context across conversations

### Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer                                │
│                    (FastAPI + SSE Streaming)                     │
├─────────────────────────────────────────────────────────────────┤
│                      LangGraph Engine                            │
│  ┌──────────┐   ┌────────────┐   ┌──────────┐   ┌───────────┐  │
│  │Initialize│──►│ Supervisor │──►│ Execute  │──►│  Output   │  │
│  │  Node    │   │    Node    │   │   Node   │   │   Node    │  │
│  └──────────┘   └────────────┘   └──────────┘   └───────────┘  │
├─────────────────────────────────────────────────────────────────┤
│     Operators          │      Tools         │       MCP         │
│   (LLM/Hybrid/MCP)     │  (Local/Remote)    │   (External)      │
├─────────────────────────────────────────────────────────────────┤
│  Documents  │  Context  │  Events  │  Cognition  │   Config    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before starting, ensure you understand:

1. **Python 3.11+** - The codebase uses modern Python features
2. **Async/await** - The entire system is asynchronous
3. **TypedDict** - Used for LangGraph state definitions
4. **Pydantic** - Used for data validation and settings
5. **FastAPI basics** - For the API layer

### Key Technologies

| Technology | Purpose |
|------------|---------|
| LangGraph | Graph-based agent orchestration |
| FastAPI | REST API and SSE streaming |
| Anthropic Claude | LLM for reasoning and generation |
| MCP | Model Context Protocol for external tools |
| Pydantic | Data validation and settings |
| hyx | Fault tolerance patterns |

---

## Phase 1: Foundation Layer

**Goal:** Set up the foundational modules that everything else depends on.

**Why first?** These modules have no internal dependencies and provide the basic building blocks (settings, data models, logging, exceptions) that all other modules need.

### Files to Create

```
src/agentic_chatbot/
├── config/
│   ├── __init__.py
│   ├── settings.py      # Environment settings (API keys, paths)
│   ├── models.py        # ModelConfig, TokenUsage, ThinkingConfig
│   └── prompts.py       # LLM prompt templates
├── data/
│   ├── __init__.py
│   ├── content.py       # ContentBlock, TextContent (atomic units)
│   ├── sourced.py       # SourcedContent with provenance
│   ├── execution.py     # ExecutionInput, ExecutionOutput
│   └── directive.py     # Directive (supervisor decision model)
├── core/
│   ├── __init__.py
│   └── exceptions.py    # Custom exception types
└── utils/
    ├── __init__.py
    └── logging.py       # Logging configuration
```

### Key Concepts

#### 1. Settings (config/settings.py)

Use Pydantic BaseSettings for environment-based configuration:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM Configuration
    anthropic_api_key: str = ""
    default_model: str = "claude-sonnet-4-20250514"

    # MCP Configuration
    mcp_discovery_url: str = ""

    # Document Storage
    document_storage_path: str = "./storage/documents"

    class Config:
        env_file = ".env"
```

#### 2. Model Registry (config/models.py)

Centralized model configuration with aliases:

```python
@dataclass
class ModelConfig:
    id: str                          # Full model ID
    name: str                        # Human-readable name
    aliases: list[str]               # Short aliases (e.g., ["sonnet"])
    supports_thinking: bool = False  # Extended thinking support

class ModelRegistry:
    _models: dict[str, ModelConfig] = {}

    @classmethod
    def register(cls, config: ModelConfig):
        cls._models[config.id] = config
        for alias in config.aliases:
            cls._models[alias] = config

    @classmethod
    def get(cls, name: str) -> ModelConfig:
        return cls._models.get(name)
```

#### 3. Token Usage Tracking (config/models.py)

Track token consumption across LLM calls:

```python
@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0      # Extended thinking
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.thinking_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            thinking_tokens=self.thinking_tokens + other.thinking_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )
```

#### 4. Content Blocks (data/content.py)

Atomic units of information:

```python
@dataclass
class TextContent:
    text: str
    content_type: str = "text"

@dataclass
class ContentBlock:
    contents: list[TextContent]
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### 5. Directives (data/directive.py)

Supervisor decisions:

```python
class DirectiveType(str, Enum):
    ANSWER = "answer"
    CALL_TOOL = "call_tool"
    CREATE_WORKFLOW = "create_workflow"
    CLARIFY = "clarify"

@dataclass
class Directive:
    type: DirectiveType
    reasoning: str
    tool_name: str | None = None
    tool_params: dict[str, Any] | None = None
    answer: str | None = None
```

### Checkpoint

Verify Phase 1 is complete:
```python
# test_phase1.py
from agentic_chatbot.config import Settings, ModelRegistry, TokenUsage
from agentic_chatbot.data import ContentBlock, TextContent, Directive, DirectiveType
from agentic_chatbot.core.exceptions import AgenticChatbotError

# Test settings
settings = Settings()
print(f"Default model: {settings.default_model}")

# Test token usage arithmetic
usage1 = TokenUsage(input_tokens=100, output_tokens=50)
usage2 = TokenUsage(input_tokens=200, output_tokens=100)
combined = usage1 + usage2
print(f"Combined tokens: {combined.total}")  # Should be 450

# Test directive creation
directive = Directive(
    type=DirectiveType.ANSWER,
    reasoning="User asked a simple question",
    answer="Hello! I'm here to help."
)
print(f"Directive type: {directive.type}")

print("✅ Phase 1 complete - Foundation layer working")
```

**What you can do now:** Nothing user-facing yet. This phase establishes the data structures and configuration that all other phases depend on.

---

## Phase 2: LLM Integration

**Goal:** Create a unified interface for calling LLMs with multi-provider support.

**Why now?** This phase depends only on Phase 1 (config, logging). All subsequent phases that need LLM calls will use this layer.

### Files to Create

```
src/agentic_chatbot/utils/
├── providers/
│   ├── __init__.py
│   ├── base.py          # BaseLLMProvider abstract class
│   ├── anthropic.py     # Anthropic direct API provider
│   └── bedrock.py       # AWS Bedrock provider
├── llm.py               # LLMClient wrapper
└── structured_llm.py    # Structured output with Pydantic validation
```

### Key Concepts

#### 1. Base Provider (utils/providers/base.py)

Abstract interface for LLM providers:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    content: str
    usage: TokenUsage
    thinking_content: str | None = None
    stop_reason: str | None = None

class BaseLLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        enable_thinking: bool = False,
        thinking_budget: int = 10000,
    ) -> LLMResponse:
        pass
```

#### 2. Anthropic Provider (utils/providers/anthropic.py)

Direct Anthropic API with extended thinking:

```python
import anthropic

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        model = kwargs.get("model", "claude-sonnet-4-20250514")
        enable_thinking = kwargs.get("enable_thinking", False)

        messages = [{"role": "user", "content": prompt}]

        # Add thinking block if enabled
        if enable_thinking:
            response = await self.client.messages.create(
                model=model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 4096),
                thinking={
                    "type": "enabled",
                    "budget_tokens": kwargs.get("thinking_budget", 10000)
                }
            )
        else:
            response = await self.client.messages.create(
                model=model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 4096),
            )

        return self._parse_response(response)
```

#### 3. LLM Client (utils/llm.py)

Unified client that selects provider based on configuration:

```python
class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._provider = self._create_provider()

    def _create_provider(self) -> BaseLLMProvider:
        if self.settings.llm_provider == "bedrock":
            return BedrockProvider(...)
        return AnthropicProvider(self.settings.anthropic_api_key)

    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        return await self._provider.complete(prompt, **kwargs)
```

#### 4. Structured LLM Caller (utils/structured_llm.py)

Pydantic validation with token tracking:

```python
from pydantic import BaseModel

@dataclass
class StructuredResult[T]:
    data: T
    usage: TokenUsage
    thinking_content: str | None = None

class StructuredLLMCaller:
    def __init__(self, client: LLMClient):
        self.client = client

    async def call_with_usage[T: BaseModel](
        self,
        prompt: str,
        response_model: type[T],
        **kwargs
    ) -> StructuredResult[T]:
        # Instruct model to return JSON
        json_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{response_model.model_json_schema()}"

        response = await self.client.complete(json_prompt, **kwargs)

        # Parse and validate
        data = response_model.model_validate_json(response.content)

        return StructuredResult(
            data=data,
            usage=response.usage,
            thinking_content=response.thinking_content
        )
```

### Checkpoint

Verify Phase 2 is complete:
```python
# test_phase2.py
import asyncio
from agentic_chatbot.utils.llm import LLMClient
from agentic_chatbot.config import Settings

async def test_llm():
    settings = Settings()

    # Verify API key is set
    if not settings.anthropic_api_key:
        print("⚠️ Set ANTHROPIC_API_KEY in .env file")
        return

    client = LLMClient(settings)

    # Test basic completion
    print("Testing LLM completion...")
    response = await client.complete("Say 'Hello World' and nothing else.")
    print(f"Response: {response.content}")
    print(f"Tokens used: {response.usage.total}")

    # Test structured output
    from pydantic import BaseModel
    from agentic_chatbot.utils.structured_llm import StructuredLLMCaller

    class SimpleAnswer(BaseModel):
        answer: str
        confidence: float

    caller = StructuredLLMCaller(client)
    result = await caller.call_with_usage(
        prompt="What is 2+2? Respond with answer and confidence (0-1).",
        response_model=SimpleAnswer
    )
    print(f"Structured answer: {result.data.answer}")
    print(f"Confidence: {result.data.confidence}")

    print("✅ Phase 2 complete - LLM integration working")

asyncio.run(test_llm())
```

**What you can do now:** You can have conversations with Claude! Try different prompts and see structured outputs. This is the first phase with a real interactive feature.

---

## Phase 3: Events & Messaging

**Goal:** Create an event system for real-time updates to clients.

**Why now?** Events are a cross-cutting concern. Nodes, operators, and MCP all emit events. Building this early allows other phases to emit events as they're built.

### Files to Create

```
src/agentic_chatbot/events/
├── __init__.py
├── types.py         # EventType enum
├── models.py        # Event data classes
└── emitter.py       # EventEmitter
```

### Key Concepts

#### 1. Event Types (events/types.py)

Enumerate all possible events:

```python
from enum import Enum

class EventType(str, Enum):
    # Supervisor events
    SUPERVISOR_THINKING = "supervisor.thinking"
    SUPERVISOR_DECIDED = "supervisor.decided"

    # Tool execution
    TOOL_START = "tool.start"
    TOOL_PROGRESS = "tool.progress"
    TOOL_COMPLETE = "tool.complete"
    TOOL_ERROR = "tool.error"

    # Response streaming
    RESPONSE_CHUNK = "response.chunk"
    RESPONSE_DONE = "response.done"

    # Direct response (operator bypasses writer)
    DIRECT_RESPONSE_START = "direct.response.start"
    DIRECT_RESPONSE_CHUNK = "direct.response.chunk"
    DIRECT_RESPONSE_DONE = "direct.response.done"

    # Elicitation (user input requests)
    ELICITATION_REQUEST = "elicitation.request"
    ELICITATION_RESPONSE = "elicitation.response"
```

#### 2. Event Models (events/models.py)

Data classes for each event type:

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class BaseEvent:
    type: EventType
    request_id: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class SupervisorThinkingEvent(BaseEvent):
    type: EventType = EventType.SUPERVISOR_THINKING
    message: str = ""

    @classmethod
    def create(cls, message: str, request_id: str | None = None):
        return cls(message=message, request_id=request_id)

@dataclass
class ToolStartEvent(BaseEvent):
    type: EventType = EventType.TOOL_START
    tool_name: str = ""

@dataclass
class ResponseChunkEvent(BaseEvent):
    type: EventType = EventType.RESPONSE_CHUNK
    content: str = ""
```

#### 3. Event Emitter (events/emitter.py)

Routes events to SSE queue and handlers:

```python
import asyncio
from typing import Callable

class EventEmitter:
    def __init__(
        self,
        queue: asyncio.Queue | None = None,
        handlers: list[Callable] | None = None
    ):
        self.queue = queue
        self.handlers = handlers or []

    async def emit(self, event: BaseEvent):
        # Send to SSE queue
        if self.queue:
            await self.queue.put(event.to_dict())

        # Call registered handlers
        for handler in self.handlers:
            await handler(event)
```

### Checkpoint

Verify Phase 3 is complete:
```python
# test_phase3.py
import asyncio
from agentic_chatbot.events import EventEmitter, EventType
from agentic_chatbot.events.models import (
    SupervisorThinkingEvent,
    ToolStartEvent,
    ResponseChunkEvent
)

async def test_events():
    # Create event queue (simulates SSE queue)
    queue = asyncio.Queue()
    emitter = EventEmitter(queue=queue)

    # Simulate a conversation flow with events
    print("Simulating event flow...")

    await emitter.emit(SupervisorThinkingEvent.create(
        "Analyzing your request...",
        request_id="test-123"
    ))

    await emitter.emit(ToolStartEvent(
        tool_name="web_search",
        request_id="test-123"
    ))

    await emitter.emit(ResponseChunkEvent(
        content="Here is the answer...",
        request_id="test-123"
    ))

    # Read events from queue
    print("\nEvents captured:")
    while not queue.empty():
        event = await queue.get()
        print(f"  - {event['type']}: {event}")

    print("\n✅ Phase 3 complete - Event system working")

asyncio.run(test_events())
```

**What you can do now:** You have a working event system. While not user-facing on its own, you can simulate and observe event flows that will power real-time updates in the final application.

---

## Phase 4: MCP Integration

**Goal:** Enable communication with external tool servers via Model Context Protocol.

**Why now?** MCP provides the infrastructure for remote tools. Operators (Phase 6) will use MCP to call external services.

### Files to Create

```
src/agentic_chatbot/mcp/
├── __init__.py
├── models.py        # Tool schemas, MessagingCapabilities
├── client.py        # MCPClient for tool invocation
├── manager.py       # Connection pooling and concurrency
├── registry.py      # MCPServerRegistry for discovery
├── callbacks.py     # Callback handlers (progress, elicitation)
└── session.py       # Session management with cleanup
```

### Key Concepts

#### 1. MCP Models (mcp/models.py)

Tool schemas and messaging capabilities:

```python
from enum import Enum
from dataclasses import dataclass

class OutputDataType(str, Enum):
    TEXT = "text"
    HTML = "html"
    IMAGE = "image"
    WIDGET = "widget"
    JSON = "json"

@dataclass
class MessagingCapabilities:
    output_types: list[OutputDataType] = field(default_factory=lambda: [OutputDataType.TEXT])
    supports_progress: bool = False
    supports_elicitation: bool = False
    supports_direct_response: bool = False
    supports_streaming: bool = False

    @classmethod
    def default(cls) -> "MessagingCapabilities":
        return cls()

@dataclass
class ToolSchema:
    name: str
    description: str
    input_schema: dict[str, Any]
    messaging: MessagingCapabilities = field(default_factory=MessagingCapabilities.default)
```

#### 2. MCP Client (mcp/client.py)

Tool invocation with callbacks:

```python
from mcp import ClientSession

class MCPClient:
    def __init__(self, session: ClientSession):
        self._session = session

    async def call_tool(
        self,
        name: str,
        params: dict[str, Any],
        callbacks: "MCPCallbacks | None" = None
    ) -> ToolResult:
        try:
            response = await self._session.call_tool(name, params)
            return self._parse_result(response)
        except Exception as e:
            if callbacks and callbacks.on_error:
                await callbacks.on_error(str(e))
            raise
```

#### 3. MCP Callbacks (mcp/callbacks.py)

Handle progress, elicitation, and content events:

```python
from typing import Callable, Awaitable

MCPProgressCallback = Callable[[str, float | None], Awaitable[None]]
MCPElicitationCallback = Callable[[str, dict], Awaitable[dict | None]]
MCPContentCallback = Callable[[str, str], Awaitable[None]]
MCPErrorCallback = Callable[[str], Awaitable[None]]

@dataclass
class MCPCallbacks:
    on_progress: MCPProgressCallback | None = None
    on_elicitation: MCPElicitationCallback | None = None
    on_content: MCPContentCallback | None = None
    on_error: MCPErrorCallback | None = None

def create_mcp_callbacks(
    emitter: EventEmitter,
    elicitation_manager: "ElicitationManager",
    request_id: str | None = None
) -> MCPCallbacks:
    async def on_progress(message: str, progress: float | None):
        await emitter.emit(ToolProgressEvent(
            message=message,
            progress=progress,
            request_id=request_id
        ))

    return MCPCallbacks(
        on_progress=on_progress,
        # ... other callbacks
    )
```

#### 4. MCP Registry (mcp/registry.py)

Service discovery for MCP servers:

```python
class MCPServerRegistry:
    def __init__(self, discovery_url: str | None = None):
        self._servers: dict[str, MCPServerConfig] = {}
        self._discovery_url = discovery_url

    async def refresh(self):
        """Fetch server configurations from discovery service."""
        if self._discovery_url:
            async with httpx.AsyncClient() as client:
                response = await client.get(self._discovery_url)
                servers = response.json()
                self._servers = {s["name"]: MCPServerConfig(**s) for s in servers}

    async def get_tools(self) -> list[ToolSchema]:
        """Get all available tools from all servers."""
        tools = []
        for server in self._servers.values():
            client = await self._get_client(server)
            server_tools = await client.list_tools()
            tools.extend(server_tools)
        return tools
```

### Checkpoint

Verify Phase 4 is complete:
```python
# test_phase4.py
import asyncio
from agentic_chatbot.mcp.models import (
    MessagingCapabilities,
    ToolSchema,
    OutputDataType
)
from agentic_chatbot.mcp import MCPServerRegistry, MCPCallbacks

async def test_mcp():
    # Test 1: MCP models work correctly
    print("Testing MCP models...")

    capabilities = MessagingCapabilities(
        output_types=[OutputDataType.TEXT, OutputDataType.HTML],
        supports_progress=True,
        supports_elicitation=True
    )
    print(f"Capabilities: progress={capabilities.supports_progress}")

    tool_schema = ToolSchema(
        name="test_tool",
        description="A test tool",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        messaging=capabilities
    )
    print(f"Tool schema: {tool_schema.name}")

    # Test 2: Registry works (without external server)
    print("\nTesting registry (offline mode)...")
    registry = MCPServerRegistry()  # No discovery URL = offline mode

    # You can manually register servers for testing
    # registry.register_server(MCPServerConfig(...))

    print("Registry initialized (no external servers configured)")

    # Test 3: If you have an MCP server running, uncomment:
    # registry = MCPServerRegistry(discovery_url="http://localhost:8080/mcp")
    # await registry.refresh()
    # tools = await registry.get_tools()
    # print(f"Available MCP tools: {[t.name for t in tools]}")

    print("\n✅ Phase 4 complete - MCP infrastructure ready")
    print("   Note: External MCP servers can be connected later")

asyncio.run(test_mcp())
```

**What you can do now:** The MCP infrastructure is ready. You can connect to external tool servers later by configuring `MCP_DISCOVERY_URL` in your `.env` file. For now, the system works without external tools.

---

## Phase 5: Document Management

**Goal:** Enable users to upload documents that provide conversation context.

**Why now?** Documents are independent of the main graph flow. Building this now provides context enrichment capability for supervisors.

### Files to Create

```
src/agentic_chatbot/documents/
├── __init__.py
├── models.py        # DocumentMetadata, DocumentStatus, DocumentChunk
├── config.py        # ChunkConfig, DocumentConfig
├── storage/
│   ├── __init__.py
│   ├── base.py      # Abstract DocumentStorage interface
│   └── local.py     # Local filesystem implementation
├── chunker.py       # Semantic document splitting
├── summarizer.py    # LLM-based summarization
├── processor.py     # Async processing pipeline
└── service.py       # High-level DocumentService API
```

### Key Concepts

#### 1. Document Models (documents/models.py)

```python
from enum import Enum

class DocumentStatus(str, Enum):
    UPLOADING = "uploading"
    CHUNKING = "chunking"
    SUMMARIZING = "summarizing"
    READY = "ready"
    ERROR = "error"

@dataclass
class DocumentMetadata:
    document_id: str
    conversation_id: str
    filename: str
    content_type: str
    status: DocumentStatus
    created_at: datetime
    chunk_count: int = 0
    summary: str | None = None

@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    content: str
    index: int
    summary: str | None = None
```

#### 2. Chunker (documents/chunker.py)

Semantic document splitting:

```python
@dataclass
class ChunkConfig:
    chunk_size: int = 4000        # ~1000 tokens per chunk
    chunk_overlap: int = 500      # Overlap to preserve context
    min_chunk_size: int = 500
    split_patterns: tuple = ("\n\n\n", "\n\n", "\n", ". ", " ")

class DocumentChunker:
    def __init__(self, config: ChunkConfig):
        self.config = config

    def chunk_document(self, content: str) -> list[str]:
        chunks = []
        current_pos = 0

        while current_pos < len(content):
            # Find best split point
            end_pos = min(current_pos + self.config.chunk_size, len(content))

            if end_pos < len(content):
                # Find natural break point
                for pattern in self.config.split_patterns:
                    split_pos = content.rfind(pattern, current_pos, end_pos)
                    if split_pos > current_pos:
                        end_pos = split_pos + len(pattern)
                        break

            chunks.append(content[current_pos:end_pos])
            current_pos = end_pos - self.config.chunk_overlap

        return chunks
```

#### 3. Document Service (documents/service.py)

High-level API:

```python
class DocumentService:
    def __init__(
        self,
        storage: DocumentStorage,
        processor: DocumentProcessor,
        llm_client: LLMClient
    ):
        self._storage = storage
        self._processor = processor
        self._llm = llm_client

    async def upload(
        self,
        conversation_id: str,
        filename: str,
        content: bytes
    ) -> DocumentMetadata:
        """Upload and start async processing."""
        doc_id = str(uuid.uuid4())

        # Save raw content
        metadata = await self._storage.save(
            document_id=doc_id,
            conversation_id=conversation_id,
            filename=filename,
            content=content
        )

        # Start background processing
        asyncio.create_task(
            self._processor.process(conversation_id, doc_id)
        )

        return metadata

    async def get_summaries(self, conversation_id: str) -> list[DocumentSummary]:
        """Get summaries for supervisor context."""
        return await self._storage.get_summaries(conversation_id)
```

### Checkpoint

Verify Phase 5 is complete:
```python
# test_phase5.py
import asyncio
import tempfile
import os
from agentic_chatbot.documents.models import DocumentStatus, DocumentMetadata
from agentic_chatbot.documents.config import ChunkConfig
from agentic_chatbot.documents.chunker import DocumentChunker
from agentic_chatbot.documents.storage.local import LocalDocumentStorage
from agentic_chatbot.documents import DocumentService, DocumentProcessor
from agentic_chatbot.utils.llm import LLMClient
from agentic_chatbot.config import Settings

async def test_documents():
    # Test 1: Chunker works correctly
    print("Testing document chunker...")

    chunker = DocumentChunker(ChunkConfig(chunk_size=100, chunk_overlap=20))

    sample_text = """
    This is the first paragraph of a test document.
    It contains multiple sentences.

    This is the second paragraph.
    It also has some content to test chunking.

    And here is a third paragraph for good measure.
    """

    chunks = chunker.chunk_document(sample_text)
    print(f"Document split into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk)} chars")

    # Test 2: Local storage works
    print("\nTesting document storage...")

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalDocumentStorage(base_path=tmpdir)

        # Save a document
        metadata = await storage.save(
            document_id="doc-123",
            conversation_id="conv-456",
            filename="test.txt",
            content=b"Hello, this is a test document for chunking and summarization."
        )
        print(f"Saved document: {metadata.document_id}")
        print(f"Status: {metadata.status}")

        # List documents
        docs = await storage.list_documents("conv-456")
        print(f"Documents in conversation: {len(docs)}")

    # Test 3: Full service (requires LLM for summarization)
    print("\nTesting full document service...")
    settings = Settings()

    if settings.anthropic_api_key:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalDocumentStorage(base_path=tmpdir)
            llm_client = LLMClient(settings)
            processor = DocumentProcessor(storage, chunker, llm_client)
            service = DocumentService(storage, processor, llm_client)

            # Upload a document
            metadata = await service.upload(
                conversation_id="conv-789",
                filename="sample.txt",
                content=b"Python is a programming language. It is widely used for web development, data science, and automation."
            )
            print(f"Uploaded: {metadata.filename}")

            # Wait briefly for background processing
            await asyncio.sleep(2)

            # Check status
            updated = await storage.get_metadata(metadata.document_id)
            print(f"Processing status: {updated.status}")
    else:
        print("⚠️ Skipping service test (no API key)")

    print("\n✅ Phase 5 complete - Document management working")

asyncio.run(test_documents())
```

**What you can do now:** You can upload documents, chunk them, and (with an API key) get AI-generated summaries. These summaries will help the supervisor make informed decisions about when to load document content.

---

## Phase 6: Operators & Tools

**Goal:** Create the pluggable action system that the supervisor can invoke.

**Why now?** Operators depend on MCP (Phase 4) and LLM (Phase 2). Tools provide local zero-latency operations.

### Files to Create

```
src/agentic_chatbot/operators/
├── __init__.py
├── base.py          # BaseOperator abstract class
├── context.py       # OperatorContext, MessagingContext
├── registry.py      # OperatorRegistry (Factory pattern)
├── llm/             # Pure LLM operators
│   ├── __init__.py
│   ├── query_rewriter.py
│   ├── writer.py
│   └── synthesizer.py
├── mcp/             # MCP-backed operators
│   ├── __init__.py
│   ├── web_searcher.py
│   └── rag_retriever.py
└── hybrid/          # LLM + MCP operators
    ├── __init__.py
    └── coder.py

src/agentic_chatbot/tools/
├── __init__.py
├── base.py          # LocalTool abstract class
├── registry.py      # LocalToolRegistry
├── provider.py      # UnifiedToolProvider
└── builtin/
    ├── __init__.py
    ├── self_info.py
    ├── capabilities.py
    └── introspection.py
```

### Key Concepts

#### 1. Base Operator (operators/base.py)

Abstract class with messaging capabilities:

```python
from abc import ABC, abstractmethod

class BaseOperator(ABC):
    name: str
    description: str

    # Messaging capabilities
    output_types: list[OutputDataType] = [OutputDataType.TEXT]
    supports_progress: bool = False
    supports_elicitation: bool = False
    supports_direct_response: bool = False
    supports_streaming: bool = False

    @abstractmethod
    async def execute(self, context: "OperatorContext") -> "OperatorResult":
        pass
```

#### 2. Operator Context (operators/context.py)

Execution context with messaging:

```python
@dataclass
class MessagingContext:
    emitter: EventEmitter
    request_id: str | None = None

    async def send_progress(self, message: str, progress: float | None = None):
        await self.emitter.emit(ToolProgressEvent(
            message=message,
            progress=progress,
            request_id=self.request_id
        ))

    async def send_content(self, content: str, direct_response: bool = False):
        if direct_response:
            await self.emitter.emit(DirectResponseChunkEvent(
                content=content,
                request_id=self.request_id
            ))

@dataclass
class OperatorContext:
    query: str
    params: dict[str, Any]
    messaging: MessagingContext | None = None
    mcp_client: MCPClient | None = None
    llm_client: LLMClient | None = None
```

#### 3. Operator Registry (operators/registry.py)

Factory pattern for operator discovery:

```python
class OperatorRegistry:
    _operators: dict[str, type[BaseOperator]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(operator_cls: type[BaseOperator]):
            cls._operators[name] = operator_cls
            return operator_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseOperator] | None:
        return cls._operators.get(name)

    @classmethod
    def list_all(cls) -> list[str]:
        return list(cls._operators.keys())
```

#### 4. Example Operator (operators/mcp/web_searcher.py)

```python
@OperatorRegistry.register("web_searcher")
class WebSearcherOperator(BaseOperator):
    name = "web_searcher"
    description = "Search the web for information"
    supports_progress = True

    async def execute(self, context: OperatorContext) -> OperatorResult:
        query = context.params.get("query", context.query)

        # Send progress
        if context.messaging:
            await context.messaging.send_progress(f"Searching for: {query}")

        # Call MCP tool
        result = await context.mcp_client.call_tool("brave_search", {"query": query})

        return OperatorResult(
            status=OperatorStatus.SUCCESS,
            contents=[TextContent(text=result.content)]
        )
```

#### 5. Local Tool (tools/base.py)

Zero-latency in-process tools:

```python
class LocalTool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any] = {}
    messaging: MessagingCapabilities = MessagingCapabilities.default()
    needs_introspection: bool = False  # Needs access to registries

    @abstractmethod
    async def execute(self, context: "LocalToolContext") -> ToolResult:
        pass
```

#### 6. Unified Tool Provider (tools/provider.py)

Merges local and remote tools:

```python
class UnifiedToolProvider:
    def __init__(
        self,
        local_registry: type[LocalToolRegistry],
        mcp_registry: MCPServerRegistry | None,
        operator_registry: type[OperatorRegistry] | None
    ):
        self._local = local_registry
        self._mcp = mcp_registry
        self._operators = operator_registry

    async def get_all_summaries(self) -> list[ToolSummary]:
        summaries = []

        # Local tools
        for name in self._local.list_all():
            tool = self._local.get(name)()
            summaries.append(ToolSummary(
                name=tool.name,
                description=tool.description,
                source="local"
            ))

        # MCP tools
        if self._mcp:
            mcp_tools = await self._mcp.get_tools()
            summaries.extend([ToolSummary(
                name=t.name,
                description=t.description,
                source="mcp"
            ) for t in mcp_tools])

        return summaries

    def is_local_tool(self, name: str) -> bool:
        return self._local.get(name) is not None
```

### Checkpoint

Verify Phase 6 is complete:
```python
# test_phase6.py
import asyncio
from agentic_chatbot.operators import OperatorRegistry
from agentic_chatbot.operators.base import BaseOperator, OperatorResult, OperatorStatus
from agentic_chatbot.operators.context import OperatorContext, MessagingContext
from agentic_chatbot.tools import UnifiedToolProvider, LocalToolRegistry
from agentic_chatbot.tools.base import LocalTool, ToolResult, ToolStatus
from agentic_chatbot.data.content import TextContent
from agentic_chatbot.events import EventEmitter
from agentic_chatbot.utils.llm import LLMClient
from agentic_chatbot.config import Settings

# Create a test operator
@OperatorRegistry.register("echo")
class EchoOperator(BaseOperator):
    name = "echo"
    description = "Echoes the input back"
    supports_progress = True

    async def execute(self, context: OperatorContext) -> OperatorResult:
        message = context.params.get("message", "No message provided")

        if context.messaging:
            await context.messaging.send_progress("Processing echo...")

        return OperatorResult(
            status=OperatorStatus.SUCCESS,
            contents=[TextContent(text=f"Echo: {message}")]
        )

# Create a test local tool
@LocalToolRegistry.register("calculator")
class CalculatorTool(LocalTool):
    name = "calculator"
    description = "Performs basic math"

    async def execute(self, context) -> ToolResult:
        expression = context.params.get("expression", "0")
        try:
            result = eval(expression)  # Simple eval for demo
            return ToolResult(
                status=ToolStatus.SUCCESS,
                contents=[TextContent(text=f"Result: {result}")]
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=str(e)
            )

async def test_operators_and_tools():
    settings = Settings()

    # Test 1: Operator registry
    print("Testing operator registry...")
    print(f"Registered operators: {OperatorRegistry.list_all()}")

    # Test 2: Execute an operator
    print("\nTesting operator execution...")
    queue = asyncio.Queue()
    emitter = EventEmitter(queue=queue)

    operator = EchoOperator()
    context = OperatorContext(
        query="test query",
        params={"message": "Hello from test!"},
        messaging=MessagingContext(emitter=emitter, request_id="test-123")
    )
    result = await operator.execute(context)
    print(f"Operator result: {result.contents[0].text}")

    # Check for progress events
    while not queue.empty():
        event = await queue.get()
        print(f"Event emitted: {event['type']}")

    # Test 3: Local tool registry
    print("\nTesting local tools...")
    print(f"Registered local tools: {LocalToolRegistry.list_all()}")

    # Test 4: Unified provider
    print("\nTesting unified tool provider...")
    provider = UnifiedToolProvider(
        local_registry=LocalToolRegistry,
        mcp_registry=None,  # No MCP for this test
        operator_registry=OperatorRegistry
    )

    summaries = await provider.get_all_summaries()
    print(f"Total tools available: {len(summaries)}")
    for s in summaries:
        print(f"  - {s.name}: {s.description} (source: {s.source})")

    # Test 5: Check if tool is local
    print(f"\nIs 'calculator' local? {provider.is_local_tool('calculator')}")
    print(f"Is 'web_search' local? {provider.is_local_tool('web_search')}")

    print("\n✅ Phase 6 complete - Operators and tools working")

asyncio.run(test_operators_and_tools())
```

**What you can do now:** You can create and execute operators and tools! The `EchoOperator` and `CalculatorTool` above are simple examples. In the real system, operators like `WebSearcherOperator` will call MCP tools, and local tools like `self_info` provide instant responses.

---

## Phase 7: Context Management

**Goal:** Optimize context assembly for supervisor prompts and manage data provenance.

**Why now?** Context management depends on data models (Phase 1) and will be used by graph nodes (Phase 8).

### Files to Create

```
src/agentic_chatbot/context/
├── __init__.py
├── models.py        # DataChunk, DataSummary, TaskContext
├── summarizer.py    # Inline summarization for large results
├── assembler.py     # Context assembly for supervisor
└── memory.py        # Memory-based context management
```

### Key Concepts

#### 1. Context Models (context/models.py)

```python
@dataclass
class DataChunk:
    """Raw data with source tracking."""
    content: str
    source: str
    source_id: int
    chunk_index: int = 0
    total_chunks: int = 1

@dataclass
class DataSummary:
    """Summarized data for supervisor decisions."""
    summary: str
    source: str
    source_id: int
    relevance_score: float = 1.0

@dataclass
class TaskContext:
    """Assembled context for a task."""
    query: str
    chunks: list[DataChunk]
    summaries: list[DataSummary]
    document_summaries: list[str]

    def to_prompt_text(self) -> str:
        """Format for supervisor prompt."""
        parts = []

        if self.summaries:
            parts.append("## Data Summaries")
            for s in self.summaries:
                parts.append(f"[Source {s.source_id}] {s.summary}")

        if self.document_summaries:
            parts.append("## Document Context")
            parts.extend(self.document_summaries)

        return "\n\n".join(parts)
```

#### 2. Context Summarizer (context/summarizer.py)

Inline summarization for large results:

```python
class ContextSummarizer:
    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    async def summarize_if_needed(
        self,
        content: str,
        max_length: int = 2000
    ) -> str:
        """Summarize content if it exceeds max length."""
        if len(content) <= max_length:
            return content

        response = await self._llm.complete(
            prompt=f"Summarize the following content concisely:\n\n{content}",
            model="haiku",  # Fast model for summarization
            max_tokens=500
        )
        return response.content
```

#### 3. Context Assembler (context/assembler.py)

Assemble context for supervisor:

```python
class ContextAssembler:
    def __init__(
        self,
        summarizer: ContextSummarizer,
        max_context_tokens: int = 8000
    ):
        self._summarizer = summarizer
        self._max_tokens = max_context_tokens

    async def assemble(
        self,
        query: str,
        tool_results: list[ToolResult],
        document_summaries: list[DocumentSummary]
    ) -> TaskContext:
        """Assemble context with token budget management."""
        chunks = []
        summaries = []
        source_id = 1

        for result in tool_results:
            content = result.get_text()

            # Summarize if too large
            if len(content) > 2000:
                summary = await self._summarizer.summarize_if_needed(content)
                summaries.append(DataSummary(
                    summary=summary,
                    source=result.tool_name,
                    source_id=source_id
                ))
            else:
                chunks.append(DataChunk(
                    content=content,
                    source=result.tool_name,
                    source_id=source_id
                ))

            source_id += 1

        return TaskContext(
            query=query,
            chunks=chunks,
            summaries=summaries,
            document_summaries=[d.summary for d in document_summaries]
        )
```

### Checkpoint

Verify Phase 7 is complete:
```python
# test_phase7.py
import asyncio
from agentic_chatbot.context.models import DataChunk, DataSummary, TaskContext
from agentic_chatbot.context.summarizer import ContextSummarizer
from agentic_chatbot.context.assembler import ContextAssembler
from agentic_chatbot.utils.llm import LLMClient
from agentic_chatbot.config import Settings

async def test_context():
    settings = Settings()

    # Test 1: Context models work
    print("Testing context models...")

    chunk = DataChunk(
        content="Python is a high-level programming language.",
        source="web_search",
        source_id=1
    )
    print(f"Created chunk from source: {chunk.source}")

    summary = DataSummary(
        summary="Programming language information",
        source="web_search",
        source_id=1,
        relevance_score=0.95
    )
    print(f"Created summary with relevance: {summary.relevance_score}")

    # Test 2: TaskContext formatting
    print("\nTesting TaskContext formatting...")

    task_context = TaskContext(
        query="What is Python?",
        chunks=[chunk],
        summaries=[summary],
        document_summaries=["User uploaded a Python tutorial document."]
    )

    prompt_text = task_context.to_prompt_text()
    print("Generated prompt context:")
    print("-" * 40)
    print(prompt_text)
    print("-" * 40)

    # Test 3: Context summarizer (requires LLM)
    if settings.anthropic_api_key:
        print("\nTesting context summarizer...")
        llm_client = LLMClient(settings)
        summarizer = ContextSummarizer(llm_client)

        long_content = """
        Python is a high-level, general-purpose programming language.
        Its design philosophy emphasizes code readability with the use of
        significant indentation. Python is dynamically typed and garbage-collected.
        It supports multiple programming paradigms, including structured,
        object-oriented and functional programming. Python was conceived in
        the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica
        (CWI) in the Netherlands as a successor to the ABC programming language,
        which was inspired by SETL, capable of exception handling and
        interfacing with the Amoeba operating system.
        """ * 3  # Make it long enough to trigger summarization

        result = await summarizer.summarize_if_needed(long_content, max_length=200)
        print(f"Original length: {len(long_content)}")
        print(f"Summarized length: {len(result)}")
        print(f"Summary: {result[:100]}...")

        # Test 4: Full assembler
        print("\nTesting context assembler...")

        from agentic_chatbot.operators.base import OperatorResult, OperatorStatus
        from agentic_chatbot.data.content import TextContent

        # Simulate tool results
        tool_results = [
            OperatorResult(
                status=OperatorStatus.SUCCESS,
                contents=[TextContent(text="Python is great for beginners.")]
            )
        ]

        assembler = ContextAssembler(summarizer)
        assembled = await assembler.assemble(
            query="What is Python?",
            tool_results=tool_results,
            document_summaries=[]
        )
        print(f"Assembled {len(assembled.chunks)} chunks, {len(assembled.summaries)} summaries")
    else:
        print("\n⚠️ Skipping LLM-based tests (no API key)")

    print("\n✅ Phase 7 complete - Context management working")

asyncio.run(test_context())
```

**What you can do now:** You can assemble and format context for the supervisor. The context assembler intelligently summarizes large tool results and formats everything for the LLM prompt. This is the preparation layer for the graph nodes.

---

## Phase 8: Graph & Nodes

**Goal:** Build the core LangGraph state machine that orchestrates the entire flow.

**Why now?** This is the core of the system. It depends on all previous phases (operators, tools, context, events).

### Files to Create

```
src/agentic_chatbot/graph/
├── __init__.py
├── state.py         # ChatState TypedDict with reducers
├── nodes.py         # Node functions (deprecated, use nodes/)
└── builder.py       # Graph construction

src/agentic_chatbot/nodes/
├── __init__.py
├── context/         # Context-related nodes
│   ├── __init__.py
│   ├── init_node.py
│   └── fetch_tools_node.py
├── orchestration/   # Decision-making nodes
│   ├── __init__.py
│   ├── supervisor_node.py
│   └── reflect_node.py
├── execution/       # Tool execution nodes
│   ├── __init__.py
│   ├── tool_node.py
│   └── synthesize_node.py
└── output/          # Response nodes
    ├── __init__.py
    ├── write_node.py
    └── stream_node.py
```

### Key Concepts

#### 1. Chat State (graph/state.py)

TypedDict with reducers for accumulating data:

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

def reduce_token_usage(left: TokenUsage | None, right: TokenUsage | None) -> TokenUsage:
    """Accumulate token usage across LLM calls."""
    if left is None:
        return right or TokenUsage()
    if right is None:
        return left
    return left + right

def reduce_tool_results(left: list | None, right: list | None) -> list:
    """Append new tool results."""
    return list(left or []) + list(right or [])

class ChatState(TypedDict, total=False):
    # Input
    user_query: str
    conversation_id: str
    request_id: str
    requested_model: str | None

    # Conversation
    messages: Annotated[list, add_messages]

    # Supervisor
    current_decision: Directive | None
    iteration: int
    max_iterations: int

    # Execution
    tool_results: Annotated[list[ToolResult], reduce_tool_results]
    current_tool: str | None

    # Context
    task_context: TaskContext | None
    document_summaries: list[DocumentSummary]

    # Token tracking
    token_usage: Annotated[TokenUsage, reduce_token_usage]

    # Output
    final_response: str | None

    # Runtime (not persisted)
    event_emitter: EventEmitter | None
    tool_provider: UnifiedToolProvider | None
```

#### 2. Initialize Node (nodes/context/init_node.py)

```python
async def initialize_node(state: ChatState) -> dict[str, Any]:
    """Validate input and set up initial state."""
    user_query = state.get("user_query", "")

    if not user_query.strip():
        raise ValueError("Empty user query")

    # Add user message to history
    user_message = HumanMessage(content=user_query)

    return {
        "messages": [user_message],
        "iteration": 0,
        "max_iterations": 5,
    }
```

#### 3. Supervisor Node (nodes/orchestration/supervisor_node.py)

The brain of the system:

```python
async def supervisor_node(state: ChatState) -> dict[str, Any]:
    """Decide what action to take."""
    emitter = state.get("event_emitter")
    tool_provider = state.get("tool_provider")

    # Emit thinking event
    if emitter:
        await emitter.emit(SupervisorThinkingEvent.create(
            "Analyzing your request...",
            request_id=state.get("request_id")
        ))

    # Get available tools
    tools_text = tool_provider.get_tools_text() if tool_provider else ""

    # Build prompt
    prompt = SUPERVISOR_PROMPT.format(
        query=state.get("user_query"),
        conversation_history=format_messages(state.get("messages", [])),
        tools=tools_text,
        tool_results=format_tool_results(state.get("tool_results", [])),
        document_context=format_documents(state.get("document_summaries", []))
    )

    # Call LLM with structured output
    caller = StructuredLLMCaller(llm_client)
    result = await caller.call_with_usage(
        prompt=prompt,
        response_model=SupervisorDecision,
        model="thinking",
        enable_thinking=True,
        thinking_budget=10000
    )

    # Emit decision event
    if emitter:
        await emitter.emit(SupervisorDecidedEvent.create(
            action=result.data.action,
            request_id=state.get("request_id")
        ))

    return {
        "current_decision": result.data,
        "iteration": state.get("iteration", 0) + 1,
        "token_usage": result.usage
    }
```

#### 4. Execute Tool Node (nodes/execution/tool_node.py)

```python
async def execute_tool_node(state: ChatState) -> dict[str, Any]:
    """Execute the selected tool or operator."""
    decision = state.get("current_decision")
    tool_provider = state.get("tool_provider")
    emitter = state.get("event_emitter")

    tool_name = decision.tool_name
    params = decision.tool_params or {}

    # Emit start event
    if emitter:
        await emitter.emit(ToolStartEvent(
            tool_name=tool_name,
            request_id=state.get("request_id")
        ))

    # Check if local tool (zero latency)
    if tool_provider.is_local_tool(tool_name):
        result = await tool_provider.execute(tool_name, params)
    else:
        # Execute as operator
        operator_cls = OperatorRegistry.get(tool_name)
        if operator_cls:
            operator = operator_cls()
            context = OperatorContext(
                query=state.get("user_query"),
                params=params,
                messaging=create_messaging_context(emitter, state.get("request_id"))
            )
            result = await operator.execute(context)
        else:
            result = ToolResult(status=ToolStatus.ERROR, error=f"Unknown tool: {tool_name}")

    # Emit complete event
    if emitter:
        await emitter.emit(ToolCompleteEvent(
            tool_name=tool_name,
            request_id=state.get("request_id")
        ))

    return {
        "tool_results": [result],
        "current_tool": tool_name
    }
```

#### 5. Graph Builder (graph/builder.py)

Wire everything together:

```python
from langgraph.graph import StateGraph, START, END

def route_supervisor_decision(state: ChatState) -> str:
    decision = state.get("current_decision")
    if not decision:
        return "clarify"

    match decision.type:
        case DirectiveType.ANSWER:
            return "write"
        case DirectiveType.CALL_TOOL:
            return "execute_tool"
        case DirectiveType.CLARIFY:
            return "clarify"
        case _:
            return "write"

def route_reflection(state: ChatState) -> str:
    # Check if operator sent direct response
    if state.get("sent_direct_response"):
        return "direct_response"

    reflection = state.get("reflection")
    if reflection and reflection.needs_more:
        # Check iteration limit
        if state.get("iteration", 0) >= state.get("max_iterations", 5):
            return "satisfied"
        return "need_more"

    return "satisfied"

def create_chat_graph() -> StateGraph:
    builder = StateGraph(ChatState)

    # Add nodes
    builder.add_node("initialize", initialize_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("execute_tool", execute_tool_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("write", write_node)
    builder.add_node("stream", stream_node)
    builder.add_node("clarify", clarify_node)

    # Add edges
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "supervisor")

    # Conditional edges from supervisor
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor_decision,
        {
            "write": "write",
            "execute_tool": "execute_tool",
            "clarify": "clarify"
        }
    )

    # Tool execution flow
    builder.add_edge("execute_tool", "reflect")

    builder.add_conditional_edges(
        "reflect",
        route_reflection,
        {
            "satisfied": "synthesize",
            "need_more": "supervisor",
            "blocked": "write",
            "direct_response": "stream"
        }
    )

    builder.add_edge("synthesize", "write")
    builder.add_edge("write", "stream")
    builder.add_edge("clarify", "stream")
    builder.add_edge("stream", END)

    return builder.compile()
```

### Graph Flow Diagram

```
START
  │
  ▼
┌─────────────┐
│ initialize  │
└─────────────┘
  │
  ▼
┌─────────────┐
│ supervisor  │◄─────────────────────────┐
└─────────────┘                          │
  │                                      │
  ├── "answer" ──► write ──► stream ──► END
  │
  ├── "call_tool" ──► execute_tool ──► reflect
  │                                      │
  │         ┌────────────────────────────┼────────┐
  │         │                            │        │
  │         ▼                            │        ▼
  │    "satisfied"                  "need_more"  "direct_response"
  │         │                            │        │
  │         ▼                            │        │
  │    synthesize ──► write ──► stream ─┴─► END ◄┘
  │
  └── "clarify" ──► clarify ──► stream ──► END
```

### Checkpoint

Verify Phase 8 is complete:
```python
# test_phase8.py
import asyncio
from agentic_chatbot.graph import create_chat_graph
from agentic_chatbot.graph.state import ChatState, create_initial_state
from agentic_chatbot.events import EventEmitter
from agentic_chatbot.tools import UnifiedToolProvider, LocalToolRegistry
from agentic_chatbot.operators import OperatorRegistry
from agentic_chatbot.config import Settings

async def test_graph():
    settings = Settings()

    if not settings.anthropic_api_key:
        print("⚠️ ANTHROPIC_API_KEY required for Phase 8 testing")
        return

    print("Testing LangGraph execution...")

    # Create event queue to capture events
    event_queue = asyncio.Queue()
    emitter = EventEmitter(queue=event_queue)

    # Create tool provider
    provider = UnifiedToolProvider(
        local_registry=LocalToolRegistry,
        mcp_registry=None,
        operator_registry=OperatorRegistry
    )

    # Create initial state
    state = create_initial_state(
        user_query="What is 2 + 2? Just give me the number.",
        conversation_id="test-conv-123",
        request_id="test-req-456",
        event_emitter=emitter,
        tool_provider=provider
    )

    # Build and run the graph
    print("Building graph...")
    graph = create_chat_graph()

    print("Executing graph...")
    result = await graph.ainvoke(state)

    # Show results
    print("\n" + "=" * 50)
    print("EXECUTION COMPLETE")
    print("=" * 50)
    print(f"\nFinal response: {result.get('final_response', 'No response')}")
    print(f"Iterations: {result.get('iteration', 0)}")

    # Show token usage
    usage = result.get("token_usage")
    if usage:
        print(f"\nToken usage:")
        print(f"  Input: {usage.input_tokens}")
        print(f"  Output: {usage.output_tokens}")
        print(f"  Thinking: {usage.thinking_tokens}")
        print(f"  Total: {usage.total}")

    # Show captured events
    print("\nEvents captured:")
    event_count = 0
    while not event_queue.empty():
        event = await event_queue.get()
        event_count += 1
        print(f"  {event_count}. {event.get('type')}")
    print(f"Total events: {event_count}")

    print("\n✅ Phase 8 complete - Graph execution working!")

asyncio.run(test_graph())
```

**What you can do now:** You have a fully working agentic chatbot! The graph orchestrates the entire conversation flow - from receiving a query, to deciding what action to take, executing tools, and generating responses. You can test different queries and see how the supervisor makes decisions.

**Try these queries:**
- `"Hello!"` - Direct answer
- `"What is Python?"` - May use tools if configured
- `"Tell me about yourself"` - Uses self_info local tool

---

## Phase 9: Cognition (Optional)

**Goal:** Add a meta-cognitive layer for user modeling and cross-conversation memory.

**Why optional?** The chatbot works without cognition. This phase adds personalization and learning capabilities.

### Files to Create

```
src/agentic_chatbot/cognition/
├── __init__.py
├── models.py           # UserProfile, EpisodicMemory, CognitiveContext
├── config.py           # CognitionSettings
├── storage.py          # PostgreSQL persistence
├── task_queue.py       # Background task processing
├── theory_of_mind.py   # User modeling
├── episodic_memory.py  # Cross-conversation memory
├── identity.py         # Learning goals and metrics
├── meta_monitor.py     # Self-reflection
└── service.py          # CognitionService unified interface
```

### Key Concepts

#### 1. Cognition Models (cognition/models.py)

```python
@dataclass
class UserProfile:
    user_id: str
    expertise_level: str = "intermediate"  # novice, intermediate, expert
    communication_style: str = "detailed"   # concise, detailed, technical
    domain_interests: list[str] = field(default_factory=list)

    def to_context_text(self) -> str:
        return f"""User Profile:
- Expertise: {self.expertise_level}
- Style: {self.communication_style}
- Interests: {', '.join(self.domain_interests[:5])}"""

@dataclass
class EpisodicMemory:
    memory_id: str
    user_id: str
    summary: str
    topics: list[str]
    importance: float
    created_at: datetime

@dataclass
class CognitiveContext:
    user_profile: UserProfile | None = None
    relevant_memories: list[EpisodicMemory] = field(default_factory=list)

    def to_context_text(self) -> str:
        parts = []
        if self.user_profile:
            parts.append(self.user_profile.to_context_text())
        if self.relevant_memories:
            parts.append("Relevant past interactions:")
            for m in self.relevant_memories[:3]:
                parts.append(f"- {m.summary}")
        return "\n".join(parts)
```

#### 2. Cognition Service (cognition/service.py)

Unified interface with fast context loading:

```python
class CognitionService:
    def __init__(
        self,
        storage: CognitionStorage,
        task_queue: CognitionTaskQueue,
        settings: CognitionSettings
    ):
        self._storage = storage
        self._queue = task_queue
        self._settings = settings

    async def get_context(
        self,
        user_id: str,
        query: str,
        timeout_ms: int = 100
    ) -> CognitiveContext:
        """Load context within timeout (non-blocking)."""
        try:
            async with asyncio.timeout(timeout_ms / 1000):
                profile = await self._storage.get_user_profile(user_id)
                memories = await self._storage.get_relevant_memories(user_id, query)
                return CognitiveContext(
                    user_profile=profile,
                    relevant_memories=memories[:5]
                )
        except asyncio.TimeoutError:
            return CognitiveContext()  # Return empty on timeout

    async def enqueue_learning(
        self,
        user_id: str,
        conversation_id: str,
        messages: list[dict]
    ) -> str | None:
        """Enqueue background learning task."""
        return await self._queue.enqueue(
            task_type=TaskType.LEARN_FROM_CONVERSATION,
            payload={
                "user_id": user_id,
                "conversation_id": conversation_id,
                "messages": messages
            }
        )
```

#### 3. Background Task Queue (cognition/task_queue.py)

PostgreSQL-backed queue:

```python
class CognitionTaskQueue:
    def __init__(self, storage: CognitionStorage):
        self._storage = storage
        self._running = False
        self._worker_task: asyncio.Task | None = None

    async def start(self):
        """Start the background worker."""
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self):
        """Stop worker, completing current task."""
        self._running = False
        if self._worker_task:
            await self._worker_task

    async def _worker_loop(self):
        while self._running:
            task = await self._claim_next_task()
            if task:
                await self._process_task(task)
            else:
                await asyncio.sleep(1.0)  # Poll interval

    async def _claim_next_task(self) -> LearningTask | None:
        """Atomic claim using FOR UPDATE SKIP LOCKED."""
        return await self._storage.claim_next_task()
```

### Checkpoint

Verify Phase 9 is complete:
```python
# test_phase9.py
import asyncio
from agentic_chatbot.cognition.models import UserProfile, EpisodicMemory, CognitiveContext
from agentic_chatbot.cognition.config import CognitionSettings
from datetime import datetime

async def test_cognition():
    # Test 1: Cognition models (no database required)
    print("Testing cognition models...")

    profile = UserProfile(
        user_id="user-123",
        expertise_level="intermediate",
        communication_style="detailed",
        domain_interests=["python", "machine learning", "web development"]
    )
    print(f"Created user profile: {profile.user_id}")
    print(f"Expertise: {profile.expertise_level}")

    memory = EpisodicMemory(
        memory_id="mem-001",
        user_id="user-123",
        summary="User asked about Python web frameworks and preferred Flask",
        topics=["python", "flask", "web frameworks"],
        importance=0.8,
        created_at=datetime.now()
    )
    print(f"\nCreated memory: {memory.summary[:50]}...")

    # Test 2: CognitiveContext formatting
    print("\nTesting cognitive context...")

    context = CognitiveContext(
        user_profile=profile,
        relevant_memories=[memory]
    )

    context_text = context.to_context_text()
    print("Generated context for supervisor:")
    print("-" * 40)
    print(context_text)
    print("-" * 40)

    # Test 3: Settings validation
    print("\nTesting cognition settings...")
    settings = CognitionSettings()
    print(f"Cognition enabled: {settings.cognition_enabled}")
    print(f"Max memories per user: {settings.max_memories_per_user}")
    print(f"Memory TTL: {settings.memory_ttl_days} days")

    # Test 4: Full service (requires PostgreSQL)
    print("\n" + "=" * 50)
    print("NOTE: Full cognition service requires PostgreSQL")
    print("=" * 50)
    print("""
To test the full service:

1. Start PostgreSQL:
   docker run -d --name postgres -e POSTGRES_PASSWORD=secret -p 5432:5432 postgres

2. Set environment variables:
   COGNITION_ENABLED=true
   COGNITION_DB_HOST=localhost
   COGNITION_DB_PASSWORD=secret

3. The service will auto-create tables on startup.
""")

    print("✅ Phase 9 complete - Cognition models ready")
    print("   (Full service available with PostgreSQL)")

asyncio.run(test_cognition())
```

**What you can do now:** Cognition models are ready. The full service (with PostgreSQL) enables:
- **User Modeling:** Track expertise level and communication preferences
- **Episodic Memory:** Remember important interactions across conversations
- **Personalization:** Adapt responses based on user history

This phase is optional - the chatbot works without it, but personalization improves user experience.

---

## Phase 10: API Layer & Application

**Goal:** Expose the chatbot via HTTP API with SSE streaming.

**Why last?** The API layer depends on everything else. It's the entry point that ties all components together.

### Files to Create

```
src/agentic_chatbot/
├── api/
│   ├── __init__.py
│   ├── models.py        # ChatRequest, ChatResponse
│   ├── sse.py           # SSE streaming helpers
│   ├── dependencies.py  # Dependency injection
│   ├── rate_limit.py    # Rate limiting
│   └── routes.py        # HTTP endpoints
├── app.py               # Application lifecycle
└── main.py              # FastAPI entry point
```

### Key Concepts

#### 1. API Models (api/models.py)

```python
from pydantic import BaseModel

class ChatRequest(BaseModel):
    conversation_id: str
    message: str
    user_id: str | None = None
    model: str | None = None
    context: dict[str, Any] | None = None

class TokenUsageResponse(BaseModel):
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    request_id: str
    usage: TokenUsageResponse | None = None
```

#### 2. Dependencies (api/dependencies.py)

```python
from typing import Annotated
from fastapi import Depends, Request

async def get_mcp_registry(request: Request) -> MCPServerRegistry | None:
    return request.app.state.mcp_registry

async def get_tool_provider(request: Request) -> UnifiedToolProvider:
    return request.app.state.tool_provider

async def get_cognition_service(request: Request) -> CognitionService | None:
    return getattr(request.app.state, "cognition_service", None)

MCPRegistryDep = Annotated[MCPServerRegistry | None, Depends(get_mcp_registry)]
ToolProviderDep = Annotated[UnifiedToolProvider, Depends(get_tool_provider)]
CognitionServiceDep = Annotated[CognitionService | None, Depends(get_cognition_service)]
```

#### 3. Chat Endpoint (api/routes.py)

```python
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/v1")

@router.post("/chat")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    mcp_registry: MCPRegistryDep,
    tool_provider: ToolProviderDep,
    cognition_service: CognitionServiceDep
) -> StreamingResponse:
    request_id = str(uuid.uuid4())

    # Create event queue for SSE
    event_queue = asyncio.Queue()
    emitter = EventEmitter(queue=event_queue)

    # Load cognitive context (if available)
    cognitive_context = None
    if cognition_service and chat_request.user_id:
        cognitive_context = await cognition_service.get_context(
            chat_request.user_id,
            chat_request.message
        )

    # Create initial state
    state = create_initial_state(
        user_query=chat_request.message,
        conversation_id=chat_request.conversation_id,
        request_id=request_id,
        event_emitter=emitter,
        tool_provider=tool_provider,
        cognitive_context=cognitive_context,
        requested_model=chat_request.model
    )

    # Start graph execution in background
    async def run_graph():
        graph = create_chat_graph()
        result = await graph.ainvoke(state)

        # Signal completion
        await event_queue.put({"type": "response.done"})

        # Enqueue learning (non-blocking)
        if cognition_service and chat_request.user_id:
            await cognition_service.enqueue_learning(
                chat_request.user_id,
                chat_request.conversation_id,
                result.get("messages", [])
            )

    asyncio.create_task(run_graph())

    # Return SSE stream
    return StreamingResponse(
        sse_generator(event_queue),
        media_type="text/event-stream"
    )

async def sse_generator(queue: asyncio.Queue):
    """Generate SSE events from queue."""
    while True:
        event = await queue.get()

        if event.get("type") == "response.done":
            yield f"data: {json.dumps(event)}\n\n"
            break

        yield f"data: {json.dumps(event)}\n\n"
```

#### 4. Application Lifecycle (app.py)

```python
class Application:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.mcp_registry: MCPServerRegistry | None = None
        self.mcp_manager: MCPClientManager | None = None
        self.tool_provider: UnifiedToolProvider | None = None
        self.cognition_service: CognitionService | None = None

    async def startup(self):
        """Initialize all services."""
        # MCP
        if self.settings.mcp_discovery_url:
            self.mcp_registry = MCPServerRegistry(self.settings.mcp_discovery_url)
            await self.mcp_registry.refresh()
            self.mcp_manager = MCPClientManager(self.mcp_registry)

        # Tool provider
        self.tool_provider = UnifiedToolProvider(
            local_registry=LocalToolRegistry,
            mcp_registry=self.mcp_registry,
            operator_registry=OperatorRegistry
        )

        # Cognition (optional)
        if self.settings.cognition_enabled:
            storage = CognitionStorage(self.settings)
            await storage.initialize()
            task_queue = CognitionTaskQueue(storage)
            await task_queue.start()
            self.cognition_service = CognitionService(storage, task_queue)

    async def shutdown(self):
        """Cleanup all services."""
        if self.cognition_service:
            await self.cognition_service.shutdown()
        if self.mcp_manager:
            await self.mcp_manager.close_all()
```

#### 5. Main Entry Point (main.py)

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    application = Application(Settings())
    await application.startup()

    app.state.mcp_registry = application.mcp_registry
    app.state.tool_provider = application.tool_provider
    app.state.cognition_service = application.cognition_service

    yield

    # Shutdown
    await application.shutdown()

app = FastAPI(
    title="Agentic Chatbot API",
    lifespan=lifespan
)

app.include_router(routes.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Checkpoint

Verify Phase 10 is complete:

**Terminal 1 - Start the server:**
```bash
# Start the server
python -m agentic_chatbot.main
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2 - Test the API:**
```bash
# Test 1: Health check
curl http://localhost:8000/health
# Expected: {"status": "healthy"}

# Test 2: List available tools
curl http://localhost:8000/api/v1/tools
# Expected: List of tools with names and descriptions

# Test 3: Simple chat (non-streaming)
curl -X POST http://localhost:8000/api/v1/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "test-1", "message": "Hello!"}'
# Expected: JSON response with conversation_id, response, usage

# Test 4: Streaming chat (SSE)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "test-2", "message": "What is Python?"}' \
  --no-buffer
# Expected: Stream of SSE events ending with response.done
```

**Python test script:**
```python
# test_phase10.py
import asyncio
import httpx

async def test_api():
    base_url = "http://localhost:8000"

    async with httpx.AsyncClient() as client:
        # Test health
        print("Testing health endpoint...")
        resp = await client.get(f"{base_url}/health")
        print(f"Health: {resp.json()}")

        # Test tools listing
        print("\nTesting tools endpoint...")
        resp = await client.get(f"{base_url}/api/v1/tools")
        tools = resp.json()
        print(f"Available tools: {len(tools)}")

        # Test sync chat
        print("\nTesting sync chat...")
        resp = await client.post(
            f"{base_url}/api/v1/chat/sync",
            json={
                "conversation_id": "test-api",
                "message": "Say hello in exactly 3 words."
            },
            timeout=60.0
        )
        result = resp.json()
        print(f"Response: {result.get('response')}")
        print(f"Tokens: {result.get('usage', {}).get('total_tokens', 'N/A')}")

    print("\n✅ Phase 10 complete - Full API working!")

asyncio.run(test_api())
```

**What you can do now:** You have a complete production-ready API! Features include:
- **SSE Streaming:** Real-time response streaming
- **Token Tracking:** Usage statistics for billing/monitoring
- **Tool Discovery:** List available tools via API
- **Graceful Shutdown:** Completes in-flight requests before stopping

---

## Running the Application

### 1. Environment Setup

Create a `.env` file:

```env
ANTHROPIC_API_KEY=your-api-key
MCP_DISCOVERY_URL=http://localhost:8080/mcp
DOCUMENT_STORAGE_PATH=./storage/documents
COGNITION_ENABLED=false
```

### 2. Install Dependencies

```bash
pip install -e .
```

### 3. Start the Server

```bash
python -m agentic_chatbot.main
```

### 4. Test with cURL

```bash
# Simple chat
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "conv-1", "message": "What is Python?"}'

# List available tools
curl http://localhost:8000/api/v1/tools
```

---

## Next Steps

After completing all phases, consider:

1. **Add more operators** - Create domain-specific operators for your use case
2. **Customize prompts** - Tune the supervisor prompt in `config/prompts.py`
3. **Enable cognition** - Set up PostgreSQL and enable user personalization
4. **Add monitoring** - Integrate with your observability stack
5. **Deploy** - Containerize and deploy to your infrastructure

---

## Summary

| Phase | Modules | Purpose |
|-------|---------|---------|
| 1 | config/, data/, core/, utils/logging | Foundation - no dependencies |
| 2 | utils/providers/, utils/llm | LLM integration |
| 3 | events/ | Real-time event system |
| 4 | mcp/ | External tool protocol |
| 5 | documents/ | Document management |
| 6 | operators/, tools/ | Pluggable actions |
| 7 | context/ | Context optimization |
| 8 | graph/, nodes/ | Core orchestration |
| 9 | cognition/ | Meta-cognitive layer |
| 10 | api/, app.py, main.py | HTTP API |

Each phase builds on the previous ones, creating a modular and extensible agentic chatbot system.

---

## What You Can Test After Each Phase

| Phase | What's Testable | Requirements |
|-------|-----------------|--------------|
| 1 | Data models, settings, token arithmetic | Python only |
| 2 | **Chat with Claude!** Basic Q&A | API key |
| 3 | Event emission and capture | Python only |
| 4 | MCP models, registry (offline mode) | Python only |
| 5 | Document chunking, storage, summarization | API key (for summaries) |
| 6 | **Custom operators and tools!** Execute locally | API key (for LLM ops) |
| 7 | Context assembly and formatting | API key (for summarization) |
| 8 | **Full chatbot!** Graph execution with tools | API key |
| 9 | User profiles, memory models | PostgreSQL (for full service) |
| 10 | **Production API!** HTTP endpoints, SSE | API key, server running |

### Minimum Viable Product (MVP) Path

If you want to get something working quickly:

1. **Phase 1** - Foundation (required)
2. **Phase 2** - LLM (required) → **You can chat!**
3. **Phase 3** - Events (required for streaming)
4. **Phase 6** - Operators/Tools (add capabilities)
5. **Phase 8** - Graph (orchestration) → **Full agent!**
6. **Phase 10** - API (expose via HTTP) → **Production ready!**

Phases 4 (MCP), 5 (Documents), 7 (Context), and 9 (Cognition) can be added later to enhance functionality.
