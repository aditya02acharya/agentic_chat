# Agentic Chatbot Backend

A ReACT-based agentic chatbot backend built with **LangGraph v1.x**, featuring extended thinking mode, comprehensive token tracking, and MCP integration.

## Features

### Core Capabilities
- **ReACT Supervisor**: Intelligent decision-making agent with extended thinking for complex reasoning
- **Four Action Types**:
  - `ANSWER`: Direct responses for simple questions
  - `CALL_TOOL`: Single tool execution for data retrieval
  - `CREATE_WORKFLOW`: Multi-step plans for complex tasks
  - `CLARIFY`: Request clarification for ambiguous queries

### LLM Features
- **Extended Thinking Mode**: Supervisor and planner use Claude's thinking tokens for complex reasoning
- **Conditional Thinking**: Coder operator auto-enables thinking for algorithm/optimization tasks
- **Comprehensive Token Tracking**: Input, output, thinking, and cache tokens tracked across all calls
- **User Model Selection**: API accepts model preference for response generation
- **Configurable Model Registry**: Centralized model configuration with aliases

### Infrastructure
- **LangGraph v1.0**: Production-ready stateful agent framework with checkpointing
- **MCP Integration**: Full Model Context Protocol support for external tools
- **Local Tools**: Zero-latency in-process tools for self-awareness and introspection
- **UnifiedToolProvider**: Single interface for local, MCP, and operator tools
- **SSE Streaming**: Real-time progress updates via Server-Sent Events
- **Direct Response**: Operators can bypass writer to send content directly to users
- **Document Context**: Upload documents for conversation-specific context with auto-summarization

### Reliability & Fault Tolerance
- **Resilience Patterns**: Retry, circuit breaker, and timeout via [hyx](https://github.com/roma-glushko/hyx) library
- **Automatic Retry**: Transient failures retry with exponential backoff (1-60s)
- **Circuit Breakers**: Prevent cascade failures to unhealthy services
- **Timeout Protection**: 5-minute timeout on graph execution, 120s on LLM calls
- **Graceful Cleanup**: Proper resource cleanup on client disconnect
- **Error Logging**: Comprehensive error logging instead of silent failures
- **Event Queue Management**: Automatic draining of unused events

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LANGGRAPH STATE FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  START → initialize → supervisor (with thinking)                             │
│      ├── "answer" ─────────────────────────────────────┐                    │
│      ├── "call_tool" → execute_tool → reflect          │                    │
│      │                    ├── "satisfied" → synthesize  │                    │
│      │                    ├── "need_more" → supervisor  │ (loop)            │
│      │                    ├── "blocked" → handle_blocked│                    │
│      │                    └── "direct_response" ────────┼──→ stream → END   │
│      ├── "create_workflow" → plan (with thinking) →    │                    │
│      │                       execute_workflow → reflect │                    │
│      └── "clarify" → clarify                           │                    │
│                        │                               │                    │
│                        └───────────────────────────────┼──→ write → stream  │
│                                                        │     (uses requested│
│                                                        │      model)        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Install from source
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with AWS Bedrock support
pip install -e ".[bedrock]"

# Install everything
pip install -e ".[all]"
```

## Configuration

Copy `.env.example` to `.env` and configure:

### Option 1: Anthropic Direct API (Default)

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-api-key-here
```

### Option 2: AWS Bedrock

```bash
LLM_PROVIDER=bedrock
BEDROCK_REGION=us-east-1

# Authentication (choose one):
# 1. IAM Role (recommended for AWS deployments) - no additional config needed
# 2. AWS Profile
BEDROCK_PROFILE=your-profile-name
# 3. Explicit credentials
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

### Model Configuration

Models are configured in `config/models.py` with aliases:

| Alias | Model | Supports Thinking |
|-------|-------|-------------------|
| `haiku` | Claude 3.5 Haiku | No |
| `sonnet` | Claude Sonnet 4 | No |
| `opus` | Claude Opus 4 | No |
| `thinking` | Claude Sonnet 4 | Yes (default) |

## Usage

### Start the Server

```bash
# Development mode
make run

# Production mode
make run-prod

# Or directly
uvicorn agentic_chatbot.main:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

#### POST /api/v1/chat
Main chat endpoint with SSE streaming.

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv-123",
    "message": "What is Python?",
    "model": "sonnet"
  }'
```

#### POST /api/v1/chat/sync
Non-streaming chat endpoint with token usage in response.

```bash
curl -X POST http://localhost:8000/api/v1/chat/sync \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv-123",
    "message": "What is Python?",
    "model": "opus"
  }'
```

Response includes token usage:
```json
{
  "conversation_id": "conv-123",
  "response": "Python is...",
  "request_id": "req-456",
  "usage": {
    "input_tokens": 1500,
    "output_tokens": 800,
    "thinking_tokens": 5000,
    "cache_read_tokens": 200,
    "cache_write_tokens": 0,
    "total_tokens": 7500
  }
}
```

#### GET /api/v1/health
Health check endpoint.

#### GET /api/v1/tools
List available tools (local + MCP + operators).

#### POST /api/v1/elicitation/respond
Submit user response to a tool's input request.

### Document Endpoints

#### POST /api/v1/documents
Upload a document for conversation context.

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv-123",
    "filename": "report.txt",
    "content": "Document content here...",
    "content_type": "text/plain"
  }'
```

Response:
```json
{
  "document_id": "doc-abc",
  "conversation_id": "conv-123",
  "filename": "report.txt",
  "status": "uploading",
  "size_bytes": 1024,
  "message": "Document uploaded, processing started"
}
```

#### GET /api/v1/documents/{conversation_id}
List all documents for a conversation with summaries.

#### GET /api/v1/documents/{conversation_id}/{document_id}/status
Get processing status (uploading → chunking → summarizing → ready).

#### DELETE /api/v1/documents/{conversation_id}/{document_id}
Delete a document from conversation.

## Project Structure

```
src/agentic_chatbot/
├── api/                    # FastAPI routes and models
│   ├── routes.py           # HTTP endpoints with timeout protection
│   ├── models.py           # Request/Response schemas incl. TokenUsageResponse
│   ├── dependencies.py     # Dependency injection (ToolProviderDep)
│   └── sse.py              # SSE helpers with cleanup
│
├── config/                 # Configuration
│   ├── settings.py         # Environment settings
│   ├── models.py           # Model registry, ThinkingConfig, TokenUsage
│   └── prompts.py          # LLM prompt templates
│
├── tools/                  # Local tools (zero-latency)
│   ├── base.py             # LocalTool base class
│   ├── registry.py         # LocalToolRegistry
│   ├── provider.py         # UnifiedToolProvider (local + MCP)
│   └── builtin/            # Built-in tools
│       ├── self_info.py    # Bot version, capabilities
│       ├── capabilities.py # Detailed feature list
│       ├── introspection.py# list_tools, list_operators
│       └── load_document.py# Document loading tools
│
├── documents/              # Document upload and context
│   ├── models.py           # DocumentStatus, DocumentChunk, etc.
│   ├── config.py           # ChunkConfig, DocumentConfig
│   ├── chunker.py          # Semantic document chunking
│   ├── summarizer.py       # LLM-based summarization
│   ├── processor.py        # Async processing pipeline
│   ├── service.py          # High-level DocumentService
│   └── storage/            # Storage backends
│       ├── base.py         # Abstract storage interface
│       └── local.py        # Local filesystem storage
│
├── operators/              # Operators (Strategy pattern)
│   ├── base.py             # BaseOperator with messaging attributes
│   ├── context.py          # OperatorContext + MessagingContext
│   ├── registry.py         # OperatorRegistry
│   ├── hybrid/             # LLM + MCP operators (e.g., coder with thinking)
│   ├── llm/                # Pure LLM operators
│   └── mcp/                # MCP-backed operators
│
├── graph/                  # LangGraph implementation
│   ├── state.py            # ChatState with token_usage reducer
│   ├── nodes.py            # Node functions with thinking mode
│   └── builder.py          # StateGraph construction
│
├── events/                 # Event system for SSE
│   ├── emitter.py          # EventEmitter with error logging
│   ├── models.py           # Event data classes
│   └── types.py            # Event type enums
│
├── mcp/                    # MCP protocol integration
│   ├── callbacks.py        # Callbacks + ElicitationManager
│   ├── client.py           # MCP client
│   ├── manager.py          # Connection management
│   └── session.py          # Session management with cleanup
│
├── context/                # Context optimization
│   ├── models.py           # DataChunk, DataSummary, TaskContext
│   └── summarizer.py       # Inline summarization
│
├── core/                   # Core domain
│   ├── supervisor.py       # SupervisorDecision model
│   ├── workflow.py         # WorkflowDefinition
│   ├── exceptions.py       # Custom exceptions
│   └── resilience.py       # Fault tolerance patterns (hyx)
│
└── utils/                  # Utilities
    ├── llm.py              # LLMClient with thinking support
    ├── structured_llm.py   # Structured output + token tracking
    ├── providers/          # Multi-provider LLM support
    │   ├── base.py         # BaseLLMProvider + LLMResponse
    │   ├── anthropic.py    # Anthropic API with thinking
    │   └── bedrock.py      # AWS Bedrock provider
    └── logging.py          # Logging configuration
```

## Extended Thinking Mode

The supervisor and planner use Claude's extended thinking for complex reasoning:

```python
# Supervisor uses thinking mode for decision-making
result = await caller.call_with_usage(
    prompt=prompt,
    response_model=SupervisorDecision,
    model="thinking",
    enable_thinking=True,
    thinking_budget=10000,  # Max thinking tokens
)

# Access thinking content and token usage
decision = result.data
thinking_content = result.thinking_content
usage = result.usage  # TokenUsage with thinking_tokens
```

### Conditional Thinking (Coder)

The coder operator automatically enables thinking for complex tasks:

```python
# Keywords that trigger thinking mode:
COMPLEX_KEYWORDS = [
    "algorithm", "optimize", "debug", "security",
    "performance", "recursive", "data structure", ...
]

# Also enabled for:
# - Queries > 300 characters
# - Multiple requirements (bullet points)
```

## Token Tracking

Token usage is accumulated across all LLM calls:

```python
@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.thinking_tokens
```

## Local Tools

Zero-latency tools for self-awareness and document access:

| Tool | Description |
|------|-------------|
| `self_info` | Bot version, capabilities, release notes |
| `list_capabilities` | Detailed feature list |
| `list_tools` | All available tools |
| `list_operators` | Registered operators |
| `list_documents` | List uploaded documents and summaries |
| `load_document` | Load document content into context |

## Resilience Patterns

Fault tolerance for all remote calls using [hyx](https://github.com/roma-glushko/hyx):

### Patterns Applied

| Pattern | Description | Configuration |
|---------|-------------|---------------|
| **Retry** | Exponential backoff for transient failures | 3 attempts, 1-60s backoff |
| **Circuit Breaker** | Stop calling failing services | Opens after 3-5 failures, recovers in 30-60s |
| **Timeout** | Bound operation duration | 120s for LLM, 30s for MCP |

### Usage

```python
from agentic_chatbot.core.resilience import (
    llm_retry,
    llm_circuit_breaker,
    llm_timeout,
    wrap_anthropic_errors,
)

@llm_retry
@llm_circuit_breaker
@llm_timeout
@wrap_anthropic_errors
async def call_llm(...):
    ...
```

### Protected Components

- **MCP Client**: Tool calls, schema fetches, health checks
- **MCP Registry**: Discovery service refresh, tool listing
- **LLM Providers**: Anthropic and Bedrock API calls

## Development

```bash
# Run tests
make test

# Run linting
make lint

# Format code
make format

# Type checking
make typecheck
```

## Document Context

Upload documents to provide conversation-specific context. Documents are automatically chunked and summarized.

### How It Works

1. **Upload**: Client uploads document via `POST /api/v1/documents`
2. **Chunking**: Document split into overlapping chunks (4000 chars, 500 overlap)
3. **Summarization**: Each chunk summarized by LLM (haiku for speed)
4. **Ready**: Supervisor sees document summaries in context
5. **Loading**: Supervisor loads full documents when relevant to query

### Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_DOCUMENTS_PER_CONVERSATION` | 5 | Max documents per conversation |
| `MAX_DOCUMENT_SIZE_BYTES` | 1MB | Max document size |
| `CHUNK_SIZE` | 4000 | Characters per chunk |
| `CHUNK_OVERLAP` | 500 | Overlap between chunks |
| `DOCUMENT_CONTEXT_PRIORITY` | 900 | High priority in context |

### Supervisor Integration

The supervisor prompt includes document summaries and prioritizes document loading:

```
Document Context:
- Use "list_documents" to see document summaries
- Use "load_document" to load content when relevant
- Documents should be given HIGH PRIORITY
```

## Design Patterns

- **Strategy Pattern**: Operators as interchangeable algorithms
- **Factory + Registry Pattern**: Dynamic operator/tool instantiation
- **Observer Pattern**: Event-driven SSE streaming
- **Builder Pattern**: Context assembly and workflow construction
- **Chain of Responsibility**: Error handling layers
- **Mediator Pattern**: Supervisor coordinates all components
- **State Pattern**: LangGraph state management with reducers
- **Composite Pattern**: Graph with conditional edges
- **Provider Pattern**: UnifiedToolProvider merges tool sources
- **Circuit Breaker Pattern**: Prevent cascade failures via hyx
- **Retry Pattern**: Automatic recovery from transient failures

## Error Handling & Reliability

- **Resilience Patterns**: Retry, circuit breaker, timeout via hyx library
- **Timeout Protection**: Graph execution times out after 5 minutes, LLM calls after 120s
- **Circuit Breakers**: Prevent cascade failures to unhealthy MCP/LLM services
- **Automatic Retry**: Transient failures retry with exponential backoff
- **Graceful Cancellation**: Background tasks get 5 seconds to cleanup
- **Event Queue Cleanup**: Remaining events drained on disconnect
- **Error Logging**: Handler errors logged instead of silently swallowed
- **Stream Cleanup**: MCP streams properly closed on session end

## License

MIT
