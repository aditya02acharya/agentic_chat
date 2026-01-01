# Two-Level Token Tracking

Comprehensive token usage tracking system with conversation-level metrics for UI display and detailed per-call tracing for internal audit.

---

## Overview

The token tracking system operates at two levels:

1. **Level 1 - Conversation-Level Metrics (UI Display)**
   - High-level token counts for user interface
   - Tracks user input, final output, and intermediate operations separately
   - Returned in API responses for cost display

2. **Level 2 - Detailed Trace (Internal Audit)**
   - Per-LLM-call tracing via OpenTelemetry (OTEL)
   - Detailed metrics for cost analysis, debugging, and optimization
   - Includes model, operation type, latency, and full token breakdown

---

## Level 1: Conversation-Level Metrics

### Purpose
Provide clear, user-facing token usage for cost transparency and usage tracking.

### Metrics Tracked

| Metric | Description | Use Case |
|--------|-------------|----------|
| `user_input_tokens` | Tokens from user's message | Show input cost |
| `final_output_tokens` | Tokens in final response to user | Show output cost |
| `intermediate_tokens` | Tokens used in agent's internal operations | Show "thinking" cost |
| `total_tokens` | Sum of all tokens | Total cost |

### Calculation

```python
total_tokens = input_tokens + output_tokens + thinking_tokens + cache_tokens
intermediate_tokens = total_tokens - (user_input_tokens + final_output_tokens)
```

### API Response Format

```json
{
  "usage": {
    // Level 1: Conversation-level (for UI)
    "user_input_tokens": 50,
    "final_output_tokens": 200,
    "intermediate_tokens": 500,
    "total_tokens": 750,

    // Level 2: Detailed breakdown (for debugging)
    "input_tokens": 600,
    "output_tokens": 200,
    "thinking_tokens": 0,
    "cache_read_tokens": 0,
    "cache_write_tokens": 0
  }
}
```

### Example UI Display

```
Token Usage:
├─ Input (your message):      50 tokens
├─ Output (response):         200 tokens
├─ Intermediate (agent work): 500 tokens
└─ Total:                     750 tokens

Estimated Cost: $0.0075
```

---

## Level 2: Detailed OTEL Tracing

### Purpose
Provide detailed per-call metrics for:
- Internal audit and compliance
- Cost optimization and analysis
- Performance monitoring
- Debugging and troubleshooting

### Implementation Status

**Current:** Placeholder (NoOp tracer)
- Infrastructure in place
- No actual tracing until OTEL SDK installed

**To Enable:**
1. Install dependencies:
   ```bash
   pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
   ```

2. Set environment variable:
   ```bash
   export OTEL_ENABLED=true
   export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
   ```

3. Update `telemetry/tracing.py`:
   - Replace NoOpTracer with real OpenTelemetry tracer
   - Configure exporter (OTLP, Jaeger, Zipkin, etc.)
   - Add resource attributes (service name, version)

### OTEL Span Attributes

Each LLM call creates a span with these attributes:

```python
{
    # Operation context
    "llm.operation": "supervisor",           # Type of operation
    "llm.conversation_id": "conv_123",
    "llm.request_id": "req_456",

    # Model details
    "llm.model": "claude-3-5-sonnet-20241022",
    "llm.provider": "anthropic",

    # Token metrics (per-call breakdown)
    "llm.usage.input_tokens": 300,
    "llm.usage.output_tokens": 100,
    "llm.usage.thinking_tokens": 0,
    "llm.usage.cache_read_tokens": 0,
    "llm.usage.cache_write_tokens": 0,
    "llm.usage.total_tokens": 400,

    # Performance
    "llm.latency_ms": 1234.5,

    # Configuration
    "llm.thinking_enabled": false,
    "llm.streaming": false,
    "llm.temperature": 0.0,
}
```

### Operation Types

| Operation | Description | Example Usage |
|-----------|-------------|---------------|
| `supervisor` | Main decision-making agent | Choosing tools, answering directly |
| `writer` | Final response generation | Formatting response with citations |
| `synthesizer` | Combining multiple sources | Merging search results |
| `tool_call` | Tool/operator execution | Web search, RAG, calculation |
| `reflection` | Assessing if more info needed | Deciding to loop or finish |
| `planner` | Creating multi-step workflows | Breaking down complex tasks |

### Example OTEL Trace

```
Trace: Request req_456
├─ llm.supervisor (500ms, 300 tokens)
│  ├─ Model: claude-3-5-sonnet
│  └─ Decision: CALL_TOOL(web_search)
├─ llm.tool_call (1200ms, 150 tokens)
│  ├─ Model: claude-3-haiku
│  └─ Tool: web_search
├─ llm.supervisor (450ms, 200 tokens)
│  ├─ Model: claude-3-5-sonnet
│  └─ Decision: ANSWER
└─ llm.writer (800ms, 250 tokens)
   ├─ Model: claude-3-5-sonnet
   └─ Final response generated

Total: 2.95s, 900 tokens
```

### OTEL Backends Supported

- **Jaeger**: Distributed tracing UI
- **Zipkin**: APM and tracing
- **Datadog**: Full observability platform
- **New Relic**: Application monitoring
- **AWS X-Ray**: AWS-native tracing
- **Google Cloud Trace**: GCP tracing
- **Custom**: Any OTLP-compatible backend

---

## Implementation Details

### Architecture

```
User Request
    ↓
API Route
    ↓
Graph Execution
    ├─ Initialize Node (tracks user_input_tokens)
    ├─ Supervisor Node (OTEL: llm.supervisor)
    ├─ Execute Tool Node (OTEL: llm.tool_call)
    ├─ Synthesize Node (OTEL: llm.synthesizer)
    ├─ Write Node (tracks final_output_tokens, OTEL: llm.writer)
    └─ Stream Node
    ↓
API Response (Level 1 metrics)
```

### Code Locations

| Component | File | Purpose |
|-----------|------|---------|
| Token Model | `config/models.py` | `TokenUsage` dataclass with Level 1 + 2 fields |
| API Response | `api/models.py` | `TokenUsageResponse` for JSON serialization |
| OTEL Tracing | `telemetry/tracing.py` | Placeholder tracer and span attributes |
| LLM Provider | `utils/providers/anthropic.py` | OTEL tracing wrapper around API calls |
| Graph Nodes | `graph/nodes.py` | Track user_input and final_output tokens |
| API Routes | `api/routes.py` | Populate token response from state |

### Tracking Flow

1. **User Input** (`initialize_node`):
   ```python
   user_input_token_count = len(user_query) // 4
   token_usage = TokenUsage(user_input_tokens=user_input_token_count)
   ```

2. **Intermediate LLM Calls** (Supervisor, Synthesizer, etc.):
   ```python
   with trace_llm_call("supervisor", model, conv_id, req_id) as attrs:
       response = await client.complete(prompt, model=model)
       attrs.input_tokens = response.usage.input_tokens
       attrs.output_tokens = response.usage.output_tokens

   # Accumulate in state
   token_usage += TokenUsage(
       input_tokens=response.usage.input_tokens,
       output_tokens=response.usage.output_tokens,
   )
   ```

3. **Final Output** (`write_node`):
   ```python
   response = await client.complete(prompt, model=model)
   token_usage = TokenUsage(
       input_tokens=response.usage.input_tokens,
       output_tokens=response.usage.output_tokens,
       final_output_tokens=response.usage.output_tokens,  # Level 1
   )
   ```

4. **API Response**:
   ```python
   TokenUsageResponse(
       user_input_tokens=token_usage.user_input_tokens,
       final_output_tokens=token_usage.final_output_tokens,
       intermediate_tokens=token_usage.intermediate_tokens,
       total_tokens=token_usage.total_tokens,
       ...
   )
   ```

---

## Cost Calculation

### Token Pricing (Example: Claude 3.5 Sonnet)

| Token Type | Price per Million |
|-----------|-------------------|
| Input | $3.00 |
| Output | $15.00 |
| Cache Read | $0.30 |
| Cache Write | $3.75 |

### Example Calculation

```python
# Level 1 metrics from API response
user_input_tokens = 50
final_output_tokens = 200
intermediate_tokens = 500

# Cost breakdown
user_input_cost = (50 / 1_000_000) * $3.00 = $0.00015
final_output_cost = (200 / 1_000_000) * $15.00 = $0.00300
intermediate_cost = (500 / 1_000_000) * $3.00 = $0.00150

total_cost = $0.00465
```

### UI Display Recommendation

```
Your message: $0.00015
Response: $0.00300
Agent work: $0.00150
━━━━━━━━━━━━━━━━━━━
Total: $0.00465
```

---

## Migration Guide

### From Old Token Tracking

**Before:**
```python
token_usage = TokenUsage(
    input_tokens=300,
    output_tokens=100,
)
```

**After:**
```python
token_usage = TokenUsage(
    input_tokens=300,
    output_tokens=100,
    user_input_tokens=50,      # NEW: Track user input separately
    final_output_tokens=100,   # NEW: Track final output separately
)

# intermediate_tokens is calculated automatically:
# = total_tokens - (user_input_tokens + final_output_tokens)
```

### Backward Compatibility

All existing fields remain:
- `input_tokens` - Still tracks all input tokens
- `output_tokens` - Still tracks all output tokens
- `total_tokens` - Still calculated the same way

New fields are additive:
- `user_input_tokens` - Defaults to 0
- `final_output_tokens` - Defaults to 0
- `intermediate_tokens` - Computed property

---

## Testing

### Unit Tests

```python
def test_token_usage_level1():
    usage = TokenUsage(
        input_tokens=600,
        output_tokens=200,
        user_input_tokens=50,
        final_output_tokens=200,
    )

    assert usage.total_tokens == 800
    assert usage.intermediate_tokens == 550  # 800 - (50 + 200)

def test_token_usage_response():
    usage = TokenUsage(
        input_tokens=600,
        output_tokens=200,
        user_input_tokens=50,
        final_output_tokens=200,
    )

    response = TokenUsageResponse(**usage.to_dict())
    assert response.user_input_tokens == 50
    assert response.final_output_tokens == 200
    assert response.intermediate_tokens == 550
    assert response.total_tokens == 800
```

### Integration Test

```python
async def test_chat_endpoint_token_tracking():
    response = await client.post(
        "/chat",
        json={
            "conversation_id": "test",
            "message": "Hello",
        }
    )

    usage = response.json()["usage"]

    # Level 1 metrics present
    assert "user_input_tokens" in usage
    assert "final_output_tokens" in usage
    assert "intermediate_tokens" in usage

    # Calculation is correct
    assert usage["total_tokens"] == (
        usage["user_input_tokens"] +
        usage["final_output_tokens"] +
        usage["intermediate_tokens"]
    )
```

---

## Future Enhancements

1. **Real-time Cost Tracking**
   - Stream cost updates during response generation
   - Show running total in UI

2. **Budget Alerts**
   - Set per-conversation or per-user limits
   - Alert when approaching budget

3. **Cost Attribution**
   - Per-operation cost breakdown
   - Identify expensive operations

4. **Cache Optimization**
   - Track cache hit rates
   - Optimize prompts for better caching

5. **OTEL Dashboards**
   - Pre-built Grafana dashboards
   - Cost analytics and optimization recommendations

---

## Troubleshooting

### Issue: `intermediate_tokens` is negative

**Cause:** `user_input_tokens` or `final_output_tokens` not properly tracked

**Solution:** Ensure `initialize_node` and `write_node` set these fields

### Issue: OTEL traces not appearing

**Cause:** NoOp tracer is still active

**Solution:**
1. Check `OTEL_ENABLED` environment variable
2. Verify OTEL SDK is installed
3. Check exporter endpoint configuration

### Issue: Token counts don't match invoice

**Cause:** Cache tokens or thinking tokens not included

**Solution:** Use `total_tokens` which includes all token types

---

## References

- **OpenTelemetry Docs**: https://opentelemetry.io/docs/
- **Anthropic Token Counting**: https://docs.anthropic.com/en/docs/build-with-claude/token-counting
- **Semantic Conventions for LLM**: https://opentelemetry.io/docs/specs/semconv/gen-ai/llm/
