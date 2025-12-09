# Agentic Chatbot Backend

A ReACT-based agentic chatbot backend built with **LangGraph v1.x** and MCP integration.

## Features

- **ReACT Supervisor**: Intelligent decision-making agent that thinks, reasons, plans, and acts
- **Four Action Types**:
  - `ANSWER`: Direct responses for simple questions
  - `CALL_TOOL`: Single tool execution for data retrieval
  - `CREATE_WORKFLOW`: Multi-step plans for complex tasks
  - `CLARIFY`: Request clarification for ambiguous queries
- **LangGraph v1.0**: Production-ready stateful agent framework with:
  - Durable state persistence
  - Built-in checkpointing
  - Human-in-the-loop patterns via `interrupt()`
- **MCP Integration**: Full Model Context Protocol support for external tools
- **SSE Streaming**: Real-time progress updates via Server-Sent Events
- **Extensible Operators**: Easy to add new operators via Registry pattern
- **Error Handling**: 4-layer error handling (tool, operator, supervisor, LLM validation)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LANGGRAPH STATE FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  START → initialize → supervisor                                             │
│      ├── "answer" ─────────────────────────────────┐                        │
│      ├── "call_tool" → execute_tool → reflect      │                        │
│      │                    ├── "satisfied" → synthesize                       │
│      │                    ├── "need_more" → supervisor (loop)                │
│      │                    └── "blocked" → handle_blocked                     │
│      └── "clarify" → clarify                       │                        │
│                        │                           │                        │
│                        └───────────────────────────┼──→ write → stream → END │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Install from source
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Required
ANTHROPIC_API_KEY=your-api-key-here

# MCP (optional)
MCP_DISCOVERY_URL=http://localhost:8080/servers

# Other settings (see .env.example for full list)
```

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
  -d '{"conversation_id": "conv-123", "message": "What is Python?"}'
```

#### POST /api/v1/chat/sync
Non-streaming chat endpoint.

```bash
curl -X POST http://localhost:8000/api/v1/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "conv-123", "message": "What is Python?"}'
```

#### GET /api/v1/health
Health check endpoint.

#### GET /api/v1/tools
List available tools.

#### POST /api/v1/elicitation/respond
Submit user response to a tool's input request.

```bash
curl -X POST http://localhost:8000/api/v1/elicitation/respond \
  -H "Content-Type: application/json" \
  -d '{"elicitation_id": "xxx", "value": "user input"}'
```

## Project Structure

```
src/agentic_chatbot/
├── api/                    # FastAPI routes and models
├── config/                 # Settings and prompts
├── context/                # Context management (memory, results, actions)
├── core/                   # Core domain (supervisor, workflow, exceptions)
├── events/                 # Event system for SSE
├── graph/                  # LangGraph implementation
│   ├── state.py           # State definitions with TypedDict + reducers
│   ├── nodes.py           # Node functions for the graph
│   └── builder.py         # StateGraph construction
├── flows/                  # Legacy flow definitions (deprecated)
├── mcp/                    # MCP protocol integration
│   ├── callbacks.py       # Callback handlers with elicitation support
│   ├── client.py          # MCP client
│   ├── manager.py         # Connection management
│   └── session.py         # Session management
├── nodes/                  # Legacy PocketFlow nodes (deprecated)
├── operators/             # Operators (Strategy pattern)
│   ├── hybrid/            # LLM + MCP operators
│   ├── llm/               # Pure LLM operators
│   └── mcp/               # MCP-backed operators
└── utils/                 # Utilities (LLM, logging)
```

## LangGraph Integration

The chatbot uses LangGraph v1.0 for orchestration. Key concepts:

### State Management

```python
from agentic_chatbot.graph.state import ChatState, create_initial_state

# State is defined with TypedDict + Annotated reducers
class ChatState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], reduce_messages]  # Appends
    tool_results: Annotated[list[ToolResult], reduce_tool_results]
    # ... other fields
```

### Creating the Graph

```python
from agentic_chatbot.graph import create_chat_graph

# Create compiled graph
graph = create_chat_graph()

# With checkpointer for persistence
from langgraph.checkpoint.memory import MemorySaver
graph = create_chat_graph(checkpointer=MemorySaver())

# With human-in-the-loop interrupts
graph = create_chat_graph(interrupt_before=["execute_tool"])
```

### Running the Graph

```python
# Async execution
config = {"configurable": {"thread_id": "conversation-123"}}
result = await graph.ainvoke(initial_state, config)

# Streaming execution
async for event in graph.astream(initial_state, config, stream_mode="updates"):
    print(event)
```

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

## Design Patterns

- **Strategy Pattern**: Operators as interchangeable algorithms
- **Factory + Registry Pattern**: Dynamic operator instantiation
- **Observer Pattern**: Event-driven SSE streaming
- **Builder Pattern**: Context assembly and workflow construction
- **Chain of Responsibility**: Error handling layers
- **Mediator Pattern**: Supervisor coordinates all components
- **State Pattern**: LangGraph state management with reducers
- **Composite Pattern**: Graph with conditional edges

## Migration from PocketFlow

The codebase has been migrated from PocketFlow to LangGraph v1.x:

| PocketFlow | LangGraph |
|------------|-----------|
| `AsyncNode.prep_async()` | Node function receives state |
| `AsyncNode.exec_async()` | Node function logic |
| `AsyncNode.post_async()` | Node function returns state updates |
| `node1 >> node2` | `builder.add_edge("node1", "node2")` |
| `node - "action" >> next` | `builder.add_conditional_edges(...)` |
| `AsyncFlow(start=node)` | `builder.compile()` |

Legacy `flows/` and `nodes/` modules are kept for backwards compatibility but are deprecated.

## License

MIT
