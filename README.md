# Agentic Chatbot Backend

A ReACT-based agentic chatbot backend built with PocketFlow and MCP integration.

## Features

- **ReACT Supervisor**: Intelligent decision-making agent that thinks, reasons, plans, and acts
- **Four Action Types**:
  - `ANSWER`: Direct responses for simple questions
  - `CALL_TOOL`: Single tool execution for data retrieval
  - `CREATE_WORKFLOW`: Multi-step plans for complex tasks
  - `CLARIFY`: Request clarification for ambiguous queries
- **MCP Integration**: Full Model Context Protocol support for external tools
- **SSE Streaming**: Real-time progress updates via Server-Sent Events
- **Extensible Operators**: Easy to add new operators via Registry pattern
- **Error Handling**: 4-layer error handling (tool, operator, supervisor, LLM validation)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MAIN CHAT FLOW                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Init → FetchTools → Supervisor                                             │
│      ├── "answer" → Write → Stream                                          │
│      ├── "call_tool" → ToolSubFlow → Observe → Reflect                     │
│      ├── "workflow" → WorkflowSubFlow → Observe → Reflect                  │
│      └── "clarify" → Clarify → Stream                                       │
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

## Project Structure

```
src/agentic_chatbot/
├── api/                    # FastAPI routes and models
├── config/                 # Settings and prompts
├── context/                # Context management (memory, results, actions)
├── core/                   # Core domain (supervisor, workflow, exceptions)
├── events/                 # Event system for SSE
├── flows/                  # PocketFlow flow definitions
├── mcp/                    # MCP protocol integration
├── nodes/                  # PocketFlow nodes
│   ├── context/           # Context preparation nodes
│   ├── execution/         # Work execution nodes
│   ├── orchestration/     # Control flow nodes
│   ├── output/            # Response generation nodes
│   └── workflow/          # Multi-step workflow nodes
├── operators/             # Operators (Strategy pattern)
│   ├── hybrid/            # LLM + MCP operators
│   ├── llm/               # Pure LLM operators
│   └── mcp/               # MCP-backed operators
└── utils/                 # Utilities (LLM, logging)
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
- **Template Method**: Base node with extension points
- **Composite Pattern**: Flows containing sub-flows

## License

MIT
