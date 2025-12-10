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
11. [Step 8: Understanding the Flow](#step-8-understanding-the-flow)
12. [Debugging Tips](#debugging-tips)

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
│   ├── dependencies.py    # Dependency injection
│   └── sse.py             # Server-Sent Events helpers
│
├── events/                 # Event system for real-time updates
│   ├── types.py           # Event type enums
│   ├── models.py          # Event data classes
│   ├── emitter.py         # Event emission
│   └── bus.py             # Event routing
│
├── operators/              # Tool implementations
│   ├── base.py            # Base operator class
│   ├── registry.py        # Operator registry (Factory pattern)
│   ├── llm/               # Pure LLM operators
│   ├── mcp/               # MCP-backed operators
│   └── hybrid/            # Combined operators
│
├── mcp/                    # MCP Protocol integration
│   ├── callbacks.py       # Callback handlers + ElicitationManager
│   ├── client.py          # MCP client
│   ├── manager.py         # Connection management
│   └── session.py         # Session management
│
├── config/                 # Configuration
│   ├── settings.py        # Environment settings
│   └── prompts.py         # LLM prompt templates
│
└── utils/                  # Utilities
    ├── llm.py             # LLM client wrapper
    ├── structured_llm.py  # Structured output with validation
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
| Input | `user_query`, `conversation_id`, `request_id` | Initial request data |
| Conversation | `messages` | Chat history (uses reducer) |
| Supervisor | `current_decision`, `iteration`, `action_history` | Decision-making state |
| Execution | `tool_results`, `current_tool`, `workflow_steps` | Tool execution state |
| Output | `final_response`, `clarify_question` | Final output data |
| Runtime | `event_emitter`, `mcp_callbacks` | Runtime context (not persisted) |

### Helper Function

```python
initial_state = create_initial_state(
    user_query="What is Python?",
    conversation_id="conv-123",
    request_id="req-456",
    event_emitter=emitter,
    # ... other runtime context
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

def route_reflection(state: ChatState) -> Literal["satisfied", "need_more", "blocked"]:
    reflection = state.get("reflection")
    return reflection.assessment if reflection else "satisfied"
```

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
  │         └──────────────┘         │   │
  │                 │                │   │
  │                 ▼                │   │
  │         ┌──────────────┐         │   │
  │         │   reflect    │         │   │
  │         └──────────────┘         │   │
  │                 │                │   │
  │    ┌────────────┼────────────┐   │   │
  │    │            │            │   │   │
  │    ▼            ▼            ▼   │   │
  │ "satisfied" "need_more"  "blocked"   │
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
  ┌────────►│   stream     │             │
  │         └──────────────┘             │
  │                │                     │
  │                ▼                     │
  │              END                     │
  │                                      │
  └── "clarify" ─────────────────────────┘
        │
        ▼
  ┌──────────────┐
  │   clarify    │───────────────────────►
  └──────────────┘
```

### Optional: Checkpointer for Persistence

```python
from langgraph.checkpoint.memory import MemorySaver

graph = create_chat_graph(checkpointer=MemorySaver())
```

---

## Step 4: Understanding the API Layer

**File:** `src/agentic_chatbot/api/routes.py`

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

1. Create `event_queue` for SSE
2. Create `EventEmitter` connected to queue
3. Create MCP callbacks wired to emitter
4. Build initial state with `create_initial_state()`
5. Start background task to run graph
6. Return `StreamingResponse` that yields from queue

```python
async def run_graph():
    graph = create_chat_graph()
    config = {"configurable": {"thread_id": conversation_id}}
    await graph.ainvoke(initial_state, config)
```

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
    SUPERVISOR_THINKING = "supervisor.thinking"
    SUPERVISOR_DECIDED = "supervisor.decided"
    TOOL_START = "tool.start"
    TOOL_PROGRESS = "tool.progress"
    TOOL_COMPLETE = "tool.complete"
    RESPONSE_CHUNK = "response.chunk"
    RESPONSE_DONE = "response.done"
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
3. Implement `execute(context) -> OperatorResult`
4. Register with decorator

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

## Step 8: Understanding the Flow

### Complete Request Flow

```
1. HTTP Request arrives at /api/v1/chat
       │
       ▼
2. Create EventEmitter + Queue
       │
       ▼
3. Create MCP callbacks
       │
       ▼
4. Create initial ChatState
       │
       ▼
5. Start background task: graph.ainvoke(state)
       │
       ▼
6. Return StreamingResponse
       │
       ▼
7. Graph executes:
   │
   ├─► initialize_node
   │       └─► Validates input, adds user message
   │
   ├─► supervisor_node
   │       ├─► Emits "thinking" event
   │       ├─► Calls LLM for decision
   │       └─► Returns decision + emits "decided" event
   │
   ├─► [Based on decision]
   │       │
   │       ├─ ANSWER ─► write_node ─► stream_node ─► END
   │       │
   │       ├─ CALL_TOOL ─► execute_tool_node
   │       │                    └─► reflect_node
   │       │                           ├─ satisfied ─► synthesize ─► write ─► stream ─► END
   │       │                           ├─ need_more ─► supervisor (loop)
   │       │                           └─ blocked ─► handle_blocked ─► write ─► stream ─► END
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
| API | `api/routes.py` | FastAPI + SSE streaming |
| Events | `events/` | Real-time updates to client |
| Operators | `operators/` | Pluggable tools (Strategy pattern) |
| MCP | `mcp/` | External tool protocol |

Start by reading `graph/state.py`, then `graph/nodes.py`, then `graph/builder.py`. This gives you the core flow. Then explore `api/routes.py` to see how it's exposed via HTTP.
