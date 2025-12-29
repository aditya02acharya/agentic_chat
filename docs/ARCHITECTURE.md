# Agentic Chatbot Architecture Guide

A step-by-step guide to understanding the LangGraph-based agentic chatbot backend.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Step 1: Understanding the State](#step-1-understanding-the-state)
5. [Step 2: Understanding Nodes](#step-2-understanding-nodes)
6. [Step 3: Understanding the Graph Builder](#step-3-understanding-the-graph-builder)
7. [Step 4: Understanding the API Layer](#step-4-understanding-the-api-layer)
8. [Step 5: Understanding Events & SSE](#step-5-understanding-events--sse)
9. [Step 6: Understanding Operators](#step-6-understanding-operators)
10. [Step 7: Understanding MCP Integration](#step-7-understanding-mcp-integration)
11. [Step 8: Understanding Local Tools](#step-8-understanding-local-tools)
12. [Step 9: Understanding the Flow](#step-9-understanding-the-flow)
13. [Step 10: Understanding Resilience Patterns](#step-10-understanding-resilience-patterns)
14. [Step 11: Understanding Document Context](#step-11-understanding-document-context)
15. [Step 12: Understanding System 3 Cognition](#step-12-understanding-system-3-cognition)
16. [Debugging Tips](#debugging-tips)

---

## Overview

This chatbot uses **LangGraph v1.x** to orchestrate a ReACT-style agent that can:
- Answer questions directly
- Call tools/operators for data retrieval
- Create multi-step workflows
- Ask for clarification when needed

### Key Technologies

| Technology | Purpose |
|------------|---------|
| LangGraph | Graph-based agent orchestration |
| FastAPI | REST API and SSE streaming |
| Anthropic Claude | LLM for reasoning and generation |
| MCP | Model Context Protocol for external tools |
| Pydantic | Data validation and settings |
| hyx | Fault tolerance patterns (retry, circuit breaker, timeout) |

---

## Prerequisites

Before diving into the code, ensure you understand:

1. **Python async/await** - The entire codebase is async
2. **TypedDict** - Used for LangGraph state definitions
3. **Pydantic** - Used for structured LLM outputs
4. **LangGraph basics** - StateGraph, nodes, edges, reducers

---

## Project Structure

```
src/agentic_chatbot/
│
├── graph/                  # 🎯 START HERE - Core LangGraph implementation
│   ├── state.py           # State definitions (TypedDict + reducers)
│   ├── nodes.py           # Node functions (the actual logic)
│   └── builder.py         # Graph construction (wiring nodes together)
│
├── api/                    # FastAPI layer
│   ├── routes.py          # HTTP endpoints
│   ├── models.py          # Request/Response schemas
│   ├── dependencies.py    # Dependency injection (incl. ToolProviderDep)
│   └── sse.py             # Server-Sent Events helpers
│
├── events/                 # Event system for real-time updates
│   ├── types.py           # Event type enums
│   ├── models.py          # Event data classes
│   ├── emitter.py         # Event emission
│   └── bus.py             # Event routing
│
├── operators/              # Tool implementations
│   ├── base.py            # Base operator class + messaging attributes
│   ├── registry.py        # Operator registry (Factory pattern)
│   ├── context.py         # OperatorContext + MessagingContext
│   ├── llm/               # Pure LLM operators
│   ├── mcp/               # MCP-backed operators
│   └── hybrid/            # Combined operators
│
├── tools/                  # 🆕 Local tools (zero-latency, in-process)
│   ├── base.py            # LocalTool base class
│   ├── registry.py        # LocalToolRegistry (decorator pattern)
│   ├── provider.py        # UnifiedToolProvider (merges local + MCP)
│   └── builtin/           # Built-in tools
│       ├── self_info.py   # Bot version, capabilities, release notes
│       ├── capabilities.py # Detailed feature list
│       ├── introspection.py # list_tools, list_operators
│       └── load_document.py # Document loading tools
│
├── documents/              # Document upload and context management
│   ├── models.py          # DocumentStatus, DocumentChunk, DocumentSummary
│   ├── config.py          # ChunkConfig, DocumentConfig
│   ├── chunker.py         # Semantic document chunking with overlap
│   ├── summarizer.py      # LLM-based chunk and document summarization
│   ├── processor.py       # Async processing pipeline
│   ├── service.py         # High-level DocumentService API
│   └── storage/           # Storage backends (abstract + implementations)
│       ├── base.py        # Abstract DocumentStorage interface
│       └── local.py       # Local filesystem implementation
│
├── cognition/              # 🆕 System 3 Meta-Cognitive Layer
│   ├── models.py          # UserProfile, EpisodicMemory, IdentityState
│   ├── config.py          # CognitionSettings
│   ├── storage.py         # PostgreSQL storage layer
│   ├── task_queue.py      # Background task queue (1 worker)
│   ├── theory_of_mind.py  # User modeling and preferences
│   ├── episodic_memory.py # Cross-conversation memory
│   ├── identity.py        # Learning goals and metrics
│   ├── meta_monitor.py    # Self-reflection and confidence
│   └── service.py         # CognitionService (unified interface)
│
├── mcp/                    # MCP Protocol integration
│   ├── callbacks.py       # Callback handlers + ElicitationManager
│   ├── client.py          # MCP client
│   ├── manager.py         # Connection management
│   ├── models.py          # Tool schemas + MessagingCapabilities
│   ├── registry.py        # MCPServerRegistry
│   └── session.py         # Session management
│
├── context/                # Context optimization
│   ├── models.py          # DataChunk, DataSummary, TaskContext
│   └── summarizer.py      # Inline summarization (haiku)
│
├── config/                 # Configuration
│   ├── settings.py        # Environment settings
│   ├── models.py          # Model registry + TokenUsage + ThinkingConfig
│   └── prompts.py         # LLM prompt templates
│
├── core/                   # Core domain
│   ├── exceptions.py      # Custom exceptions
│   └── resilience.py      # Fault tolerance patterns (hyx)
│
└── utils/                  # Utilities
    ├── llm.py             # LLM client wrapper with thinking support
    ├── structured_llm.py  # Structured output with validation + token tracking
    ├── providers/         # Multi-provider LLM support
    │   ├── base.py        # BaseLLMProvider + LLMResponse
    │   ├── anthropic.py   # Anthropic direct API with thinking
    │   └── bedrock.py     # AWS Bedrock provider
    └── logging.py         # Logging configuration
```

---

## Step 1: Understanding the State

**File:** `src/agentic_chatbot/graph/state.py`

The state is the central data structure that flows through the entire graph. Every node reads from and writes to this state.

### Key Concept: TypedDict + Annotated Reducers

```python
class ChatState(TypedDict, total=False):
    # Simple fields (replaced on update)
    user_query: str
    current_decision: SupervisorDecision | None

    # Annotated fields (appended on update)
    messages: Annotated[list[BaseMessage], reduce_messages]
    tool_results: Annotated[list[ToolResult], reduce_tool_results]
```

### Why Reducers?

Without reducers, returning `{"messages": [new_message]}` would **replace** the entire list. With `Annotated[..., reducer]`, new values are **merged** using the reducer function.

```python
def reduce_messages(left, right):
    """Appends new messages to existing list."""
    return list(left or []) + list(right or [])
```

### State Sections

| Section | Fields | Purpose |
|---------|--------|---------|
| Input | `user_query`, `conversation_id`, `request_id`, `requested_model` | Initial request data |
| Conversation | `messages` | Chat history (uses reducer) |
| Supervisor | `current_decision`, `iteration`, `action_history`, `current_task_context` | Decision-making state |
| Execution | `tool_results`, `current_tool`, `workflow_steps` | Tool execution state |
| Context | `data_chunks`, `data_summaries`, `source_counter` | Context optimization (citations) |
| Messaging | `sent_direct_response`, `direct_response_contents` | Direct response tracking |
| Token Tracking | `token_usage` | Accumulated token counts (uses reducer) |
| Output | `final_response`, `clarify_question` | Final output data |
| Runtime | `event_emitter`, `mcp_callbacks`, `tool_provider`, `elicitation_manager` | Runtime context (not persisted) |

### Token Usage Tracking

Token usage is tracked across all LLM calls using a reducer:

```python
@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0      # Extended thinking tokens
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.thinking_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        # Accumulates tokens from multiple calls
        ...

# In ChatState
token_usage: Annotated[TokenUsage, reduce_token_usage]
```

Each node that calls an LLM returns `{"token_usage": usage}` which gets accumulated via the reducer.

### Helper Function

```python
initial_state = create_initial_state(
    user_query="What is Python?",
    conversation_id="conv-123",
    request_id="req-456",
    event_emitter=emitter,
    event_queue=event_queue,
    mcp_registry=mcp_registry,
    mcp_session_manager=mcp_session_manager,
    mcp_callbacks=mcp_callbacks,
    elicitation_manager=elicitation_manager,
    tool_provider=tool_provider,  # UnifiedToolProvider for local + remote tools
    user_context=user_context,
    requested_model="sonnet",  # Model preference for response generation
)
```

---

## Step 2: Understanding Nodes

**File:** `src/agentic_chatbot/graph/nodes.py`

Nodes are async functions that:
1. Receive the current state
2. Perform some work
3. Return a partial state dict with updates

### Node Function Signature

```python
async def node_name(state: ChatState) -> dict[str, Any]:
    # Read from state
    query = state.get("user_query", "")

    # Do work...
    result = await some_operation()

    # Return updates (only changed fields)
    return {
        "some_field": result,
        "tool_results": [new_result],  # Appended due to reducer
    }
```

### Key Nodes

| Node | Purpose | Input | Output |
|------|---------|-------|--------|
| `initialize_node` | Validates input, adds user message | `user_query` | `messages` |
| `supervisor_node` | Decides action (ANSWER/CALL_TOOL/etc) | Full state | `current_decision`, `iteration` |
| `execute_tool_node` | Runs the selected operator | `current_decision` | `tool_results` |
| `reflect_node` | Evaluates tool results | `tool_results` | `reflection` |
| `synthesize_node` | Combines results into response | `tool_results` | `final_response` |
| `write_node` | Formats final output | `final_response` or `current_decision` | `final_response`, `messages` |
| `stream_node` | Streams response via SSE | `final_response` | Events emitted |
| `clarify_node` | Asks for clarification | `current_decision` | `clarify_question` |
| `handle_blocked_node` | Handles failure cases | `reflection` | `final_response` |

### Routing Functions

Routing functions determine which path to take at conditional edges:

```python
def route_supervisor_decision(state: ChatState) -> Literal["answer", "call_tool", "create_workflow", "clarify"]:
    decision = state.get("current_decision")
    return decision.action.lower() if decision else "clarify"

def route_reflection(state: ChatState) -> Literal["satisfied", "need_more", "blocked", "direct_response"]:
    # If operator sent content directly to user, skip synthesis/write
    if state.get("sent_direct_response"):
        return "direct_response"

    reflection = state.get("reflection")
    return reflection.assessment if reflection else "satisfied"
```

The `direct_response` route is used when an operator bypasses the normal response flow by sending content directly to the user (e.g., widgets, images, or streaming data).

---

## Step 3: Understanding the Graph Builder

**File:** `src/agentic_chatbot/graph/builder.py`

The builder constructs the StateGraph by adding nodes and edges.

### Basic Structure

```python
from langgraph.graph import StateGraph, START, END

def create_chat_graph():
    # 1. Create graph with state schema
    builder = StateGraph(ChatState)

    # 2. Add nodes
    builder.add_node("initialize", initialize_node)
    builder.add_node("supervisor", supervisor_node)
    # ... more nodes

    # 3. Add edges
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "supervisor")

    # 4. Add conditional edges
    builder.add_conditional_edges(
        "supervisor",           # From node
        route_supervisor_decision,  # Routing function
        {                       # Mapping
            "answer": "write",
            "call_tool": "execute_tool",
            "clarify": "clarify",
        }
    )

    # 5. Compile
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
  ├── "answer" ──────────────────────┐   │
  │                                  │   │
  ├── "call_tool" ──┐                │   │
  │                 ▼                │   │
  │         ┌──────────────┐         │   │
  │         │ execute_tool │         │   │
  │         │  (local or   │         │   │
  │         │   operator)  │         │   │
  │         └──────────────┘         │   │
  │                 │                │   │
  │                 ▼                │   │
  │         ┌──────────────┐         │   │
  │         │   reflect    │         │   │
  │         └──────────────┘         │   │
  │                 │                │   │
  │    ┌────────────┼────────────┬───────┐
  │    │            │            │   │   │
  │    ▼            ▼            ▼   │   ▼
  │ "satisfied" "need_more"  "blocked"  "direct_response"
  │    │            │            │   │   │
  │    ▼            │            ▼   │   │
  │ ┌──────────┐    │    ┌───────────┐   │
  │ │synthesize│    └───►│handle_    │   │
  │ └──────────┘         │blocked    │   │
  │    │                 └───────────┘   │
  │    │                       │         │
  │    └───────────┬───────────┘         │
  │                │                     │
  │                ▼                     │
  │         ┌──────────────┐             │
  └────────►│    write     │             │
            └──────────────┘             │
                   │                     │
                   ▼                     │
            ┌──────────────┐             │
  ┌────────►│   stream     │◄────────────┘
  │         └──────────────┘  (direct_response skips write)
  │                │
  │                ▼
  │              END
  │
  └── "clarify" ────────────┐
        │                   │
        ▼                   │
  ┌──────────────┐          │
  │   clarify    │──────────┘
  └──────────────┘
```

**Note:** The `direct_response` route allows operators to bypass the `synthesize` and `write` nodes when they've already sent content directly to the user (e.g., widgets, images, streaming responses).

### Optional: Checkpointer for Persistence

```python
from langgraph.checkpoint.memory import MemorySaver

graph = create_chat_graph(checkpointer=MemorySaver())
```

---

## Step 4: Understanding the API Layer

**File:** `src/agentic_chatbot/api/routes.py`

### Dependency Injection

FastAPI dependencies provide runtime context:

```python
# Type aliases in dependencies.py
MCPRegistryDep = Annotated[MCPServerRegistry | None, Depends(get_mcp_registry)]
MCPSessionManagerDep = Annotated[MCPSessionManager | None, Depends(get_mcp_session_manager)]
ElicitationManagerDep = Annotated[ElicitationManager, Depends(get_elicitation_manager)]
ToolProviderDep = Annotated[UnifiedToolProvider, Depends(get_tool_provider)]
```

The `ToolProviderDep` provides a unified interface to all tools (local + remote).

### Main Chat Endpoint (SSE Streaming)

```
POST /api/v1/chat
Content-Type: application/json

{
    "conversation_id": "conv-123",
    "message": "What is Python?"
}
```

**Flow:**

1. Inject dependencies (`mcp_registry`, `tool_provider`, `elicitation_manager`)
2. Create `event_queue` for SSE
3. Create `EventEmitter` connected to queue
4. Create MCP callbacks wired to emitter
5. Build initial state with `create_initial_state()` (includes `tool_provider`)
6. Start background task to run graph
7. Return `StreamingResponse` that yields from queue

```python
@router.post("/chat")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    mcp_registry: MCPRegistryDep,
    mcp_session_manager: MCPSessionManagerDep,
    elicitation_manager: ElicitationManagerDep,
    tool_provider: ToolProviderDep,  # Unified local + remote tools
) -> StreamingResponse:
    ...
```

### Tools Endpoint

```
GET /api/v1/tools
```

Lists all available tools via `UnifiedToolProvider` (local + MCP + operators).

### Sync Endpoint (Non-streaming)

```
POST /api/v1/chat/sync
```

Same as above but waits for completion and returns JSON response.

### Elicitation Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /elicitation/respond` | Submit user response to tool prompt |
| `GET /elicitation/pending` | List waiting elicitations |
| `DELETE /elicitation/{id}` | Cancel an elicitation |

---

## Step 5: Understanding Events & SSE

**Files:** `src/agentic_chatbot/events/`

Events provide real-time updates to the client via Server-Sent Events.

### Event Types

```python
class EventType(str, Enum):
    # Supervisor events
    SUPERVISOR_THINKING = "supervisor.thinking"
    SUPERVISOR_DECIDED = "supervisor.decided"

    # Tool execution events
    TOOL_START = "tool.start"
    TOOL_PROGRESS = "tool.progress"
    TOOL_COMPLETE = "tool.complete"
    TOOL_ERROR = "tool.error"

    # Response streaming
    RESPONSE_CHUNK = "response.chunk"
    RESPONSE_DONE = "response.done"

    # Direct response (operator bypasses writer)
    DIRECT_RESPONSE = "direct.response"
    DIRECT_RESPONSE_START = "direct.response.start"
    DIRECT_RESPONSE_CHUNK = "direct.response.chunk"
    DIRECT_RESPONSE_DONE = "direct.response.done"

    # Elicitation (user input requests)
    ELICITATION_REQUEST = "elicitation.request"
    ELICITATION_RESPONSE = "elicitation.response"
    # ... more
```

### Emitting Events from Nodes

```python
async def supervisor_node(state: ChatState):
    # Get emitter from state
    emitter = get_emitter(state)

    # Emit event
    await emitter.emit(
        SupervisorThinkingEvent.create(
            "Analyzing your question...",
            request_id=state.get("request_id"),
        )
    )
```

### SSE Format

```
data: {"type": "supervisor.thinking", "message": "Analyzing..."}

data: {"type": "tool.start", "tool": "web_search"}

data: {"type": "response.chunk", "content": "Python is..."}

data: {"type": "response.done"}
```

---

## Step 6: Understanding Operators

**Files:** `src/agentic_chatbot/operators/`

Operators are the tools/actions the agent can use.

### Operator Types

| Type | Description | Example |
|------|-------------|---------|
| Pure LLM | Only uses LLM | `QueryRewriterOperator` |
| MCP-backed | Calls MCP tools | `RAGRetrieverOperator` |
| Hybrid | LLM + MCP | `CoderOperator` |

### Messaging Capabilities

Operators can now communicate directly with users via `MessagingContext`:

```python
class BaseOperator:
    # Messaging capability attributes
    output_types: list[OutputDataType] = [OutputDataType.TEXT]
    supports_progress: bool = False
    supports_elicitation: bool = False
    supports_direct_response: bool = False
    supports_streaming: bool = False
```

| Capability | Purpose |
|------------|---------|
| `output_types` | Data types the operator can return (text, html, image, widget, json) |
| `supports_progress` | Can send intermediate progress updates |
| `supports_elicitation` | Can request user input during execution |
| `supports_direct_response` | Can bypass writer and send content directly |
| `supports_streaming` | Can stream content in chunks |

### Using MessagingContext

Operators receive a `MessagingContext` through their `OperatorContext`:

```python
async def execute(self, context: OperatorContext) -> OperatorResult:
    messaging = context.messaging

    # Send progress update
    await messaging.send_progress("Processing step 1 of 3...")

    # Send content directly to user (bypasses writer)
    await messaging.send_content("Here's your result", direct_response=True)

    # Send a widget (e.g., chart, interactive component)
    await messaging.send_widget("<div>...</div>", widget_type="chart")

    # Request user confirmation
    confirmed = await messaging.confirm("Proceed with deletion?")

    # Request user choice
    choice = await messaging.choose(
        "Select format:",
        options=["JSON", "CSV", "XML"]
    )
```

### Operator Registry (Factory Pattern)

```python
# Register operator
@OperatorRegistry.register("query_rewriter")
class QueryRewriterOperator(BaseOperator):
    ...

# Get operator by name
operator_cls = OperatorRegistry.get("query_rewriter")
operator = operator_cls()
result = await operator.execute(context)
```

### Creating a New Operator

1. Create file in `operators/llm/`, `operators/mcp/`, or `operators/hybrid/`
2. Inherit from `BaseOperator`
3. Set messaging capability attributes if needed
4. Implement `execute(context) -> OperatorResult`
5. Use `context.messaging` for direct user communication
6. Register with decorator

---

## Step 7: Understanding MCP Integration

**Files:** `src/agentic_chatbot/mcp/`

MCP (Model Context Protocol) enables communication with external tools.

### Callback System

```python
@dataclass
class MCPCallbacks:
    on_progress: MCPProgressCallback | None
    on_elicitation: MCPElicitationCallback | None
    on_content: MCPContentCallback | None
    on_error: MCPErrorCallback | None
```

### Elicitation Flow

When an MCP tool needs user input:

```
Tool requests input
       │
       ▼
MCPElicitationHandler creates pending request
       │
       ▼
Emits mcp.elicitation event to UI
       │
       ▼
Handler awaits Future (blocks)
       │
       ▼
User sees prompt in UI
       │
       ▼
User submits via POST /elicitation/respond
       │
       ▼
ElicitationManager resolves Future
       │
       ▼
Handler returns response to tool
       │
       ▼
Tool continues execution
```

### Creating Callbacks

```python
callbacks = create_mcp_callbacks(
    emitter=event_emitter,
    elicitation_manager=elicitation_manager,
    request_id=request_id,
)
```

---

## Step 8: Understanding Local Tools

**Files:** `src/agentic_chatbot/tools/`

Local tools provide zero-latency, in-process operations that don't require network calls.

### Why Local Tools?

| Aspect | Local Tools | Remote MCP Tools |
|--------|-------------|------------------|
| Latency | Zero (in-process) | Network dependent |
| Use Case | Self-awareness, introspection | External services |
| Execution | Synchronous | Async with callbacks |
| Context | Has access to registries | Isolated |

### Built-in Local Tools

| Tool | Purpose |
|------|---------|
| `self_info` | Returns bot version, capabilities, release notes |
| `list_capabilities` | Detailed feature list (what bot can/cannot do) |
| `list_tools` | Lists all available tools (local + remote) |
| `list_operators` | Lists all registered operators |

### LocalTool Base Class

```python
class LocalTool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any] = {}
    messaging: MessagingCapabilities = MessagingCapabilities.default()
    needs_introspection: bool = False  # Needs access to registries

    @abstractmethod
    async def execute(self, context: LocalToolContext) -> ToolResult:
        pass
```

### LocalToolRegistry (Decorator Pattern)

```python
# Register a local tool
@LocalToolRegistry.register("my_tool")
class MyTool(LocalTool):
    name = "my_tool"
    description = "Does something useful"

    async def execute(self, context: LocalToolContext) -> ToolResult:
        return ToolResult(
            status=ToolStatus.SUCCESS,
            contents=[TextContent(text="Result")]
        )

# Get tool by name
tool_cls = LocalToolRegistry.get("my_tool")
tool = tool_cls()
```

### UnifiedToolProvider

The `UnifiedToolProvider` merges local and remote tools into a single interface:

```python
class UnifiedToolProvider:
    def __init__(
        self,
        local_registry: type[LocalToolRegistry],
        mcp_registry: MCPServerRegistry | None,
        operator_registry: type[OperatorRegistry] | None,
    ):
        ...

    async def get_all_summaries(self) -> list[ToolSummary]:
        """Get summaries from all sources."""

    def is_local_tool(self, name: str) -> bool:
        """Check if tool is local (zero-latency)."""

    async def execute(self, name: str, params: dict, **kwargs) -> ToolResult:
        """Execute local tool by name."""

    def get_tools_text(self) -> str:
        """Get formatted text for supervisor prompt."""
```

### Integration with Graph

The `execute_tool_node` automatically routes to local tools:

```python
async def execute_tool_node(state: ChatState) -> dict[str, Any]:
    tool_provider = state.get("tool_provider")

    # Check if local tool (zero-latency)
    if tool_provider.is_local_tool(tool_name):
        return await _execute_local_tool(...)

    # Otherwise, execute as operator
    return await _execute_operator(...)
```

### Creating a New Local Tool

1. Create file in `tools/builtin/`
2. Inherit from `LocalTool`
3. Implement `execute(context) -> ToolResult`
4. Register with decorator
5. Import in `tools/builtin/__init__.py`

---

## Step 9: Understanding the Flow

### Complete Request Flow

```
1. HTTP Request arrives at /api/v1/chat
       │
       ▼
2. Inject dependencies (mcp_registry, tool_provider, elicitation_manager)
       │
       ▼
3. Create EventEmitter + Queue
       │
       ▼
4. Create MCP callbacks
       │
       ▼
5. Create initial ChatState (includes tool_provider)
       │
       ▼
6. Start background task: graph.ainvoke(state)
       │
       ▼
7. Return StreamingResponse
       │
       ▼
8. Graph executes:
   │
   ├─► initialize_node
   │       └─► Validates input, adds user message
   │
   ├─► supervisor_node
   │       ├─► Emits "thinking" event
   │       ├─► Gets tool list from UnifiedToolProvider
   │       ├─► Calls LLM for decision
   │       └─► Returns decision + emits "decided" event
   │
   ├─► [Based on decision]
   │       │
   │       ├─ ANSWER ─► write_node ─► stream_node ─► END
   │       │
   │       ├─ CALL_TOOL ─► execute_tool_node
   │       │                    │
   │       │                    ├─ [Local tool?] ─► _execute_local_tool (zero latency)
   │       │                    └─ [Operator?] ─► _execute_operator (may use MCP)
   │       │                           │
   │       │                           └─► reflect_node
   │       │                                  ├─ satisfied ─► synthesize ─► write ─► stream ─► END
   │       │                                  ├─ need_more ─► supervisor (loop)
   │       │                                  ├─ blocked ─► handle_blocked ─► write ─► stream ─► END
   │       │                                  └─ direct_response ─► stream ─► END (skip write)
   │       │
   │       └─ CLARIFY ─► clarify_node ─► stream_node ─► END
   │
   └─► Events streamed to client throughout
```

### Iteration Loop

The supervisor can loop up to `max_iterations` (default 5):

1. Supervisor decides CALL_TOOL
2. Tool executes
3. Reflect evaluates results
4. If "need_more" → back to supervisor
5. Repeat until satisfied, blocked, or max iterations

---

## Step 10: Understanding Resilience Patterns

**File:** `src/agentic_chatbot/core/resilience.py`

The resilience module provides fault tolerance for all remote calls using the **hyx** library.

### Why Resilience Patterns?

The chatbot makes many remote calls that can fail:
- **MCP tool calls**: External services may be unavailable
- **LLM API calls**: Rate limits, timeouts, server errors
- **Registry refresh**: Discovery service may be down

Without proper handling, a single failure can cascade and break the entire request.

### Available Patterns

| Pattern | Purpose | Use Case |
|---------|---------|----------|
| **Retry** | Automatically retry transient failures | Network timeouts, 5xx errors |
| **Circuit Breaker** | Stop calling failing services temporarily | Prevent cascade failures |
| **Timeout** | Bound operation duration | Prevent hanging requests |
| **Bulkhead** | Limit concurrent operations | Resource protection |
| **Fallback** | Graceful degradation | Return default on failure |

### Pre-configured Decorators

```python
from agentic_chatbot.core.resilience import (
    mcp_retry,
    mcp_circuit_breaker,
    llm_retry,
    llm_circuit_breaker,
    llm_timeout,
)

# MCP calls: retry + circuit breaker
@mcp_retry
@mcp_circuit_breaker
async def call_mcp_tool(...):
    ...

# LLM calls: retry + circuit breaker + timeout
@llm_retry
@llm_circuit_breaker
@llm_timeout
async def call_llm_api(...):
    ...
```

### Configuration

Centralized configuration in `ResilienceConfig`:

```python
class ResilienceConfig:
    # MCP Configuration
    MCP_RETRY_ATTEMPTS: int = 3
    MCP_RETRY_BACKOFF_BASE: float = 1.0  # seconds
    MCP_RETRY_BACKOFF_MAX: float = 30.0  # seconds
    MCP_CIRCUIT_FAILURE_THRESHOLD: int = 5
    MCP_CIRCUIT_RECOVERY_TIME: float = 30.0  # seconds

    # LLM Configuration
    LLM_RETRY_ATTEMPTS: int = 3
    LLM_RETRY_BACKOFF_BASE: float = 2.0  # longer for rate limits
    LLM_RETRY_BACKOFF_MAX: float = 60.0  # seconds
    LLM_CIRCUIT_FAILURE_THRESHOLD: int = 3
    LLM_CIRCUIT_RECOVERY_TIME: float = 60.0  # seconds
    LLM_TIMEOUT: float = 120.0  # seconds
```

### Custom Exception Types

```python
class TransientError(Exception):
    """Error likely to succeed on retry (network issues, timeouts)."""
    pass

class RateLimitError(Exception):
    """HTTP 429 or throttling errors."""
    pass
```

### Error Wrapping Decorators

Provider-specific errors are converted to resilience-aware exceptions:

```python
from agentic_chatbot.core.resilience import (
    wrap_anthropic_errors,
    wrap_aws_errors,
    wrap_httpx_errors,
)

# Anthropic API errors → TransientError/RateLimitError
@wrap_anthropic_errors
async def call_anthropic(...):
    ...

# AWS/Bedrock errors → TransientError/RateLimitError
@wrap_aws_errors
async def call_bedrock(...):
    ...

# httpx errors → TransientError/RateLimitError
@wrap_httpx_errors
async def call_http(...):
    ...
```

### Applied Locations

| Component | File | Patterns Applied |
|-----------|------|------------------|
| MCP Client | `mcp/client.py` | retry, circuit breaker |
| MCP Registry | `mcp/registry.py` | retry, circuit breaker |
| Anthropic Provider | `utils/providers/anthropic.py` | retry, circuit breaker, timeout |
| Bedrock Provider | `utils/providers/bedrock.py` | retry, circuit breaker, timeout |

### Circuit Breaker States

```
CLOSED (normal) ──[failures exceed threshold]──► OPEN (failing)
                                                      │
                    ◄──[recovery time elapsed]────────┘
                                                      │
                                                      ▼
                                                 HALF-OPEN
                                                      │
                    ◄──[success]──────────────────────┤
                    ──[failure]───────────────────────►
```

### Example: MCP Client with Resilience

```python
class MCPClient:
    @mcp_retry
    @mcp_circuit_breaker
    async def call_tool(self, name: str, params: dict) -> ToolResult:
        try:
            response = await self._session.call_tool(name, params)
            return self._parse_result(response)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(f"Rate limited: {e}") from e
            if e.response.status_code >= 500:
                raise TransientError(f"Server error: {e}") from e
            raise
        except BreakerOpen as e:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error=f"Service unavailable (circuit breaker open): {e}",
            )
```

---

## Step 11: Understanding Document Context

**Files:** `src/agentic_chatbot/documents/`

The document module enables users to upload documents that provide conversation-specific context.

### Why Document Context?

| Problem | Solution |
|---------|----------|
| Large documents don't fit in context | Chunking with summarization |
| Irrelevant documents waste tokens | Supervisor decides which to load |
| Information at chunk boundaries | Overlapping chunks preserve context |
| Slow processing blocks chat | Async background processing |

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DOCUMENT PROCESSING FLOW                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Upload ──► Storage ──► Chunker ──► Summarizer ──► Ready    │
│    │           │           │            │             │      │
│    ▼           ▼           ▼            ▼             ▼      │
│  API      metadata.json  chunks/    summary.json  Supervisor │
│           original.txt   000.json               sees summary │
│                          001.json                            │
│                          ...                                 │
│                                                              │
│  User Question ──► Supervisor ──► load_document ──► Answer   │
│                        │              │                      │
│                        ▼              ▼                      │
│                  Check summaries  Load full content          │
│                  for relevance   or specific chunks          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| **DocumentService** | `service.py` | High-level API for upload, process, load |
| **DocumentProcessor** | `processor.py` | Async pipeline (chunk → summarize → store) |
| **DocumentChunker** | `chunker.py` | Semantic splitting with overlap |
| **DocumentSummarizer** | `summarizer.py` | LLM-based chunk and document summaries |
| **DocumentStorage** | `storage/base.py` | Abstract interface for persistence |
| **LocalDocumentStorage** | `storage/local.py` | Filesystem implementation |

### Document States

```python
class DocumentStatus(str, Enum):
    UPLOADING = "uploading"     # Initial upload
    CHUNKING = "chunking"       # Being split into chunks
    SUMMARIZING = "summarizing" # Chunks being summarized
    READY = "ready"             # Fully processed, available
    ERROR = "error"             # Processing failed
```

### Processing Pipeline

```python
class DocumentProcessor:
    async def process_document(self, conversation_id: str, document_id: str):
        # 1. Update status to CHUNKING
        await self._storage.update_status(document_id, DocumentStatus.CHUNKING)

        # 2. Load and chunk content
        content = await self._storage.load_content(document_id)
        chunks = self._chunker.chunk_document(content)
        await self._storage.save_chunks(document_id, chunks)

        # 3. Update status to SUMMARIZING
        await self._storage.update_status(document_id, DocumentStatus.SUMMARIZING)

        # 4. Summarize chunks in parallel (semaphore-controlled)
        async with asyncio.Semaphore(MAX_CONCURRENT):
            summaries = await asyncio.gather(*[
                self._summarizer.summarize_chunk(chunk)
                for chunk in chunks
            ])

        # 5. Aggregate into document summary
        doc_summary = await self._summarizer.aggregate_summaries(chunks)

        # 6. Update status to READY
        await self._storage.update_status(document_id, DocumentStatus.READY)
```

### Chunking Strategy

Documents are split at semantic boundaries with overlap:

```python
@dataclass
class ChunkConfig:
    CHUNK_SIZE: int = 4000           # ~1000 tokens per chunk
    CHUNK_OVERLAP: int = 500         # Overlap to preserve context
    MIN_CHUNK_SIZE: int = 500        # Don't create tiny chunks
    MAX_CHUNKS_PER_DOC: int = 50     # Limit chunks per document
    SPLIT_PATTERNS: tuple = (        # Priority order for splits
        "\n\n\n",  # Section breaks
        "\n\n",    # Paragraphs
        "\n",      # Lines
        ". ",      # Sentences
        " ",       # Words (last resort)
    )
```

### Supervisor Integration

The supervisor prompt includes document context:

```python
SUPERVISOR_DECISION_PROMPT = """User Query: {query}

Conversation Context:
{conversation_context}

Document Context:
{document_context}  # ← Summaries injected here

Tool Results So Far:
{tool_results}
...
Remember:
- PRIORITY: Check document summaries first - if relevant documents exist,
  load them before using external tools
"""
```

### Graph State Integration

Document fields in ChatState:

```python
class ChatState(TypedDict, total=False):
    # ... existing fields ...

    # Document summaries for supervisor decision-making
    document_summaries: Annotated[list[Any], reduce_document_summaries]

    # Loaded document content for context
    loaded_documents: Annotated[list[Any], reduce_loaded_documents]
```

### Local Tools for Documents

| Tool | Purpose |
|------|---------|
| `list_documents` | Get all document summaries for conversation |
| `load_document` | Load full or partial document content |

```python
@LocalToolRegistry.register
class LoadDocumentTool(LocalTool):
    name = "load_document"
    description = "Load document content into context"

    async def execute(self, context: LocalToolContext) -> ToolResult:
        doc_service = context.document_service
        document_ids = context.params.get("document_ids", [])

        # Wait for documents if still processing
        await doc_service.wait_for_documents(document_ids)

        # Load full documents or specific chunks
        loaded = await doc_service.load_full_document(doc_id)
        return self.success(loaded.content)
```

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/documents` | POST | Upload document |
| `/documents/{conv_id}` | GET | List documents |
| `/documents/{conv_id}/{doc_id}/status` | GET | Get processing status |
| `/documents/{conv_id}/{doc_id}` | DELETE | Delete document |

### Storage Structure

```
storage/documents/{conversation_id}/{document_id}/
├── original.txt       # Raw document content
├── metadata.json      # DocumentMetadata
├── summary.json       # DocumentSummary (after processing)
└── chunks/
    ├── 000.json       # First chunk with summary
    ├── 001.json       # Second chunk with summary
    └── ...
```

### Error Handling

The processor uses resilience patterns for summarization:

```python
class DocumentSummarizer:
    @llm_retry
    @llm_circuit_breaker
    @llm_timeout
    async def summarize_chunk(self, content: str) -> ChunkSummaryResult:
        # LLM call with fault tolerance
        ...
```

### Configuration

```python
@dataclass
class DocumentConfig:
    MAX_DOCUMENTS_PER_CONVERSATION: int = 5
    MAX_DOCUMENT_SIZE_BYTES: int = 1_000_000  # 1MB
    ALLOWED_CONTENT_TYPES: tuple = ("text/plain", "text/markdown")
    SUMMARIZATION_MODEL: str = "haiku"  # Fast model for summaries
    MAX_CONCURRENT_SUMMARIZATIONS: int = 5
    DOCUMENT_CONTEXT_PRIORITY: int = 900  # High priority in context
    DEFAULT_WAIT_TIMEOUT: float = 30.0  # Max wait for processing
```

---

## Step 12: Understanding System 3 Cognition

**Files:** `src/agentic_chatbot/cognition/`

The cognition module implements a meta-cognitive layer (System 3) that provides persistent learning, user modeling, and self-reflection capabilities.

### Why System 3?

| Problem | Solution |
|---------|----------|
| No memory across conversations | Episodic memory with deduplication |
| Generic responses for all users | Theory of Mind (user modeling) |
| No learning from interactions | Background learning via task queue |
| System doesn't know itself | Identity tracking (goals, metrics) |

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SYSTEM 3 COGNITION FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Request                                                                     │
│     │                                                                        │
│     ▼                                                                        │
│  ┌─────────────────┐     Fast (<100ms)    ┌─────────────────┐               │
│  │ CognitionService│─────────────────────►│ CognitiveContext│               │
│  │  get_context()  │                      │  - UserProfile   │               │
│  └─────────────────┘                      │  - Memories      │               │
│           │                               │  - Identity      │               │
│           │                               └────────┬─────────┘               │
│           │                                        │                         │
│           │                                        ▼                         │
│           │                               ┌─────────────────┐               │
│           │                               │ Supervisor Node │               │
│           │                               │ (enriched prompt)│               │
│           │                               └────────┬─────────┘               │
│           │                                        │                         │
│           │                                        ▼                         │
│           │                               ┌─────────────────┐               │
│           │                               │    Response     │◄── Returns    │
│           │                               │                 │    immediately │
│           │                               └────────┬─────────┘               │
│           │                                        │                         │
│           ▼                                        ▼                         │
│  ┌─────────────────┐    Non-blocking     ┌─────────────────┐               │
│  │ CognitionService│◄────────────────────│  Stream Node    │               │
│  │enqueue_learning()                     │ (after response) │               │
│  └────────┬────────┘                     └─────────────────┘               │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐         ┌─────────────────┐                           │
│  │  PostgreSQL     │────────►│  Task Queue     │                           │
│  │  Task Table     │  claim  │  (1 worker)     │                           │
│  └─────────────────┘         └────────┬────────┘                           │
│                                       │                                     │
│                    ┌──────────────────┼──────────────────┐                 │
│                    ▼                  ▼                  ▼                 │
│           ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│           │Theory of Mind│  │  Episodic    │  │   Identity   │            │
│           │(user profile)│  │   Memory     │  │  (metrics)   │            │
│           └──────────────┘  └──────────────┘  └──────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Non-blocking**: Response returns immediately; learning happens in background
2. **Fast Context Loading**: Context loads within 100ms timeout
3. **Deduplication**: Similar memories are merged, not duplicated
4. **Graceful Degradation**: If cognition fails, system works without it

### Components

| Component | File | Purpose |
|-----------|------|---------|
| **CognitionService** | `service.py` | Unified interface for all operations |
| **TheoryOfMind** | `theory_of_mind.py` | User modeling (expertise, style, interests) |
| **EpisodicMemory** | `episodic_memory.py` | Cross-conversation memory with dedup |
| **Identity** | `identity.py` | Learning goals, performance metrics |
| **MetaMonitor** | `meta_monitor.py` | Confidence scoring, error patterns |
| **CognitionStorage** | `storage.py` | PostgreSQL persistence layer |
| **CognitionTaskQueue** | `task_queue.py` | Background task processing (1 worker) |

### CognitionService Interface

```python
class CognitionService:
    # Fast, synchronous - for request enrichment
    async def get_context(
        self,
        user_id: str,
        query: str,
    ) -> CognitiveContext:
        """Load context within 100ms timeout."""
        ...

    # Fast, non-blocking - for background learning
    async def enqueue_learning(
        self,
        user_id: str,
        conversation_id: str,
        messages: list[dict],
        outcome: str | None = None,
    ) -> str | None:
        """Enqueue task and return immediately."""
        ...
```

### Theory of Mind (User Modeling)

Builds user profiles from interaction patterns:

```python
@dataclass
class UserProfile:
    user_id: str
    expertise_level: str = "intermediate"  # novice, intermediate, expert
    communication_style: str = "detailed"   # concise, detailed, technical
    domain_interests: list[str] = field(default_factory=list)
    interaction_patterns: dict[str, Any] = field(default_factory=dict)
    preferences: dict[str, Any] = field(default_factory=dict)

    def to_context_text(self) -> str:
        """Format for supervisor prompt injection."""
        return f"""User Profile:
- Expertise: {self.expertise_level}
- Style preference: {self.communication_style}
- Interests: {', '.join(self.domain_interests[:5])}"""
```

Expertise is inferred from signals in messages:
- **Novice signals**: "what is", "how do I", "explain", "beginner"
- **Expert signals**: "implementation", "optimize", "architecture", technical terms

### Episodic Memory

Stores cross-conversation memories with deduplication:

```python
@dataclass
class EpisodicMemory:
    memory_id: str
    user_id: str
    conversation_id: str
    summary: str
    outcome: str  # success, partial, failure
    topics: list[str]
    importance: float  # 0-1 score
    access_count: int
    created_at: datetime
    last_accessed: datetime
```

**Deduplication**: Uses Jaccard similarity on topics:
```python
def _calculate_topic_similarity(self, topics1: list[str], topics2: list[str]) -> float:
    set1, set2 = set(topics1), set(topics2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

# If similarity >= 0.7, memories are merged instead of duplicated
```

**Pruning**: Keeps memories within limits:
- Max 100 memories per user
- 90-day TTL for old memories
- Low importance (<0.3) memories pruned first

### Background Task Queue

PostgreSQL-backed queue with atomic task claiming:

```python
class CognitionTaskQueue:
    async def enqueue(
        self,
        task_type: TaskType,
        payload: dict,
        delay_seconds: int = 0,
    ) -> str:
        """Insert task into cognition_tasks table."""
        ...

    async def _claim_next_task(self) -> LearningTask | None:
        """Atomic claim using FOR UPDATE SKIP LOCKED."""
        query = """
        UPDATE cognition_tasks
        SET status = 'processing', started_at = NOW()
        WHERE id = (
            SELECT id FROM cognition_tasks
            WHERE status = 'pending'
            AND scheduled_for <= NOW()
            ORDER BY created_at
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        RETURNING *
        """
        ...
```

**Retry Logic**: Failed tasks retry with exponential backoff (max 3 attempts).

**Graceful Shutdown**: Worker completes current task before stopping.

### PostgreSQL Schema

```sql
-- User profiles for Theory of Mind
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id VARCHAR(255) PRIMARY KEY,
    expertise_level VARCHAR(50) DEFAULT 'intermediate',
    communication_style VARCHAR(50) DEFAULT 'detailed',
    domain_interests JSONB DEFAULT '[]',
    interaction_patterns JSONB DEFAULT '{}',
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Cross-conversation episodic memories
CREATE TABLE IF NOT EXISTS episodic_memories (
    memory_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    conversation_id VARCHAR(255) NOT NULL,
    summary TEXT NOT NULL,
    outcome VARCHAR(50) NOT NULL,
    topics JSONB DEFAULT '[]',
    importance FLOAT DEFAULT 0.5,
    access_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    last_accessed TIMESTAMP DEFAULT NOW()
);

-- System identity state
CREATE TABLE IF NOT EXISTS identity_state (
    id VARCHAR(50) PRIMARY KEY DEFAULT 'singleton',
    learning_goals JSONB DEFAULT '[]',
    knowledge_gaps JSONB DEFAULT '[]',
    performance_metrics JSONB DEFAULT '{}',
    core_values JSONB DEFAULT '{}',
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Background task queue
CREATE TABLE IF NOT EXISTS cognition_tasks (
    id VARCHAR(255) PRIMARY KEY,
    task_type VARCHAR(100) NOT NULL,
    payload JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    attempts INT DEFAULT 0,
    max_attempts INT DEFAULT 3,
    scheduled_for TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT
);

-- Indexes for efficient queries
CREATE INDEX idx_memories_user_id ON episodic_memories(user_id);
CREATE INDEX idx_memories_topics ON episodic_memories USING GIN(topics);
CREATE INDEX idx_tasks_status ON cognition_tasks(status, scheduled_for);
```

### Graph Integration

The cognition service integrates with the graph at two points:

**1. Initialize Node** (context loading):
```python
async def initialize_node(state: ChatState) -> dict[str, Any]:
    # Load cognitive context if user_id provided
    cognition_service = state.get("cognition_service")
    user_id = state.get("user_id")

    if cognition_service and user_id:
        query = state.get("user_query", "")
        cognitive_context = await cognition_service.get_context(user_id, query)
        return {"cognitive_context": cognitive_context, ...}

    return {...}
```

**2. Stream Node** (learning enqueue):
```python
async def stream_node(state: ChatState) -> dict[str, Any]:
    # ... stream response ...

    # Enqueue learning (non-blocking)
    await _enqueue_cognitive_learning(state, outcome="success")

    return {...}
```

### Supervisor Prompt Injection

Cognitive context is injected into the supervisor prompt:

```python
# In supervisor_node
cognitive_text = state.get("cognitive_context", CognitiveContext()).to_context_text()

prompt = SUPERVISOR_DECISION_PROMPT.format(
    query=user_query,
    cognitive_context=cognitive_text,  # Injected here
    conversation_context=conversation_context,
    ...
)
```

The context includes:
- User profile (expertise, style, interests)
- Relevant memories from past conversations
- Identity summary (if applicable)

### Configuration

```python
class CognitionSettings(BaseSettings):
    # Enable/disable
    cognition_enabled: bool = True
    theory_of_mind_enabled: bool = True
    episodic_memory_enabled: bool = True
    identity_enabled: bool = True

    # PostgreSQL connection
    cognition_db_host: str = "localhost"
    cognition_db_port: int = 5432
    cognition_db_name: str = "cognition"
    cognition_db_user: str = "postgres"
    cognition_db_password: str = ""

    # Memory limits
    max_memories_per_user: int = 100
    memory_ttl_days: int = 90
    similarity_threshold: float = 0.7  # For deduplication
    min_importance_threshold: float = 0.3  # For pruning

    # Task queue
    task_poll_interval_seconds: float = 1.0
    task_max_retries: int = 3

    # Performance
    context_load_timeout_ms: int = 100  # Fast context loading
```

### API Changes

The chat request now accepts `user_id`:

```python
class ChatRequest(BaseModel):
    conversation_id: str
    message: str
    user_id: str | None = None  # For cognition features
    context: dict[str, Any] | None = None
    model: str | None = None
```

### FastAPI Lifespan Integration

```python
# In main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app_instance = Application()
    await app_instance.startup()
    app.state.cognition_service = app_instance.cognition_service

    yield

    # Shutdown (worker completes current task)
    await app_instance.shutdown()
```

---

## Debugging Tips

### 1. Check Events in Browser

Open browser DevTools → Network → Filter by "chat" → Look at EventStream

### 2. Add Logging to Nodes

```python
from agentic_chatbot.utils.logging import get_logger
logger = get_logger(__name__)

async def my_node(state: ChatState):
    logger.info("Entering my_node", user_query=state.get("user_query"))
    # ...
```

### 3. Inspect State at Each Node

```python
async def debug_node(state: ChatState):
    import json
    # Serialize state (excluding non-serializable)
    debug_state = {k: v for k, v in state.items()
                   if k not in ["event_emitter", "event_queue"]}
    logger.debug("State", state=debug_state)
    return {}
```

### 4. Test Individual Components

```python
# Test operator
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.operators.context import OperatorContext

operator = OperatorRegistry.get("query_rewriter")()
ctx = OperatorContext(user_query="test", params={})
result = await operator.execute(ctx)
print(result)
```

### 5. Test Graph Execution

```python
from agentic_chatbot.graph import create_chat_graph
from agentic_chatbot.graph.state import create_initial_state

graph = create_chat_graph()
state = create_initial_state(
    user_query="Hello",
    conversation_id="test",
    request_id="test-1",
)

result = await graph.ainvoke(state)
print(result.get("final_response"))
```

---

## Summary

| Component | File | Key Concept |
|-----------|------|-------------|
| State | `graph/state.py` | TypedDict + reducers |
| Nodes | `graph/nodes.py` | Async functions returning state updates |
| Graph | `graph/builder.py` | StateGraph with conditional edges |
| API | `api/routes.py` | FastAPI + SSE streaming + dependency injection |
| Events | `events/` | Real-time updates to client |
| Operators | `operators/` | Pluggable tools with messaging capabilities |
| Local Tools | `tools/` | Zero-latency in-process tools |
| Tool Provider | `tools/provider.py` | UnifiedToolProvider (local + remote) |
| MCP | `mcp/` | External tool protocol |
| Context | `context/` | DataChunk/DataSummary for citations |
| Resilience | `core/resilience.py` | Fault tolerance (retry, circuit breaker, timeout) |
| Documents | `documents/` | Document upload, chunking, summarization |
| Cognition | `cognition/` | System 3 meta-cognitive layer (user modeling, memory) |

### New Features Summary

| Feature | Description |
|---------|-------------|
| **Local Tools** | Zero-latency tools for self-awareness (`self_info`, `list_capabilities`) |
| **UnifiedToolProvider** | Merges local and remote tools into single interface |
| **MessagingContext** | Allows operators to send progress, direct responses, elicit input |
| **Direct Response** | Operators can bypass writer and send content directly to user |
| **Context Optimization** | DataChunks for raw data, DataSummaries for supervisor decisions |
| **Extended Thinking** | Supervisor, planner, and coder use thinking mode for complex reasoning |
| **Token Tracking** | Full token usage tracking across all LLM calls (input, output, thinking) |
| **Model Configuration** | Centralized model registry with aliases and thinking support |
| **Fault Tolerance** | Retry, circuit breaker, timeout patterns via hyx for all remote calls |
| **Document Context** | Upload documents with auto-summarization for conversation context |
| **System 3 Cognition** | Meta-cognitive layer with user modeling, episodic memory, and identity |

---

## Extended Thinking Mode

**Files:** `src/agentic_chatbot/config/models.py`, `src/agentic_chatbot/utils/providers/`

Extended thinking enables Claude models to "think" before responding, improving quality for complex reasoning tasks.

### Model Configuration

```python
@dataclass
class ModelConfig:
    id: str                          # Full model ID (e.g., "claude-sonnet-4-20250514")
    name: str                        # Human-readable name
    aliases: list[str]               # Short aliases (e.g., ["sonnet", "claude-sonnet"])
    supports_thinking: bool = False  # Whether model supports extended thinking
    default_thinking_budget: int = 10000

# Registry provides lookup by alias or ID
config = ModelRegistry.get("thinking")  # Returns thinking-enabled model
model_id = config.id  # "claude-sonnet-4-20250514"
```

### Using Extended Thinking

The supervisor and planner nodes automatically use extended thinking:

```python
# In supervisor_node
result = await caller.call_with_usage(
    prompt=prompt,
    response_model=SupervisorDecision,
    system=system,
    model="thinking",           # Alias for thinking-enabled model
    enable_thinking=True,
    thinking_budget=10000,      # Max thinking tokens
)

# Access thinking content and usage
decision = result.data
thinking_content = result.thinking_content  # Model's reasoning process
usage = result.usage  # TokenUsage with thinking_tokens
```

### Conditional Thinking (Coder Operator)

The coder operator conditionally uses thinking mode for complex tasks:

```python
class CoderOperator(BaseOperator):
    def _should_use_thinking(self, query: str) -> bool:
        """Detect complex tasks requiring extended thinking."""
        # Keywords indicating complexity
        COMPLEX_KEYWORDS = ["algorithm", "optimize", "debug", "security", ...]

        query_lower = query.lower()
        for keyword in COMPLEX_KEYWORDS:
            if keyword in query_lower:
                return True

        # Long queries are likely complex
        if len(query) > 300:
            return True

        return False

    async def execute(self, context: OperatorContext):
        use_thinking = self._should_use_thinking(context.query)

        if use_thinking:
            response = await client.complete(
                prompt=prompt,
                model="thinking",
                enable_thinking=True,
                thinking_budget=15000,
            )
        else:
            response = await client.complete(
                prompt=prompt,
                model="sonnet",
            )
```

### Token Tracking

Token usage is tracked comprehensively across all LLM calls:

```python
@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0      # Extended thinking tokens
    cache_read_tokens: int = 0    # Prompt caching
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.thinking_tokens
```

The API returns total token usage in the response:

```python
# POST /api/v1/chat/sync response
{
    "conversation_id": "conv-123",
    "response": "...",
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

### User Model Selection

Users can specify their preferred model for response generation:

```python
# API request
{
    "conversation_id": "conv-123",
    "message": "What is Python?",
    "model": "opus"  # User preference for writer
}

# In write_node
requested_model = state.get("requested_model") or "sonnet"
response = await client.complete(prompt, model=requested_model)
```

---

Start by reading `graph/state.py`, then `graph/nodes.py`, then `graph/builder.py`. This gives you the core flow. Then explore `api/routes.py` to see how it's exposed via HTTP. For self-awareness features, see `tools/builtin/`. For thinking mode, see `config/models.py`.
