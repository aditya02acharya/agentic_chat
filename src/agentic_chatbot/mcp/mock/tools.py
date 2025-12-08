"""Mock tool implementations for testing."""

from typing import Any


MOCK_TOOLS = {
    "web_search": {
        "name": "web_search",
        "description": "Search the web for information",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    "rag_search": {
        "name": "rag_search",
        "description": "Search internal knowledge base",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    "run_python": {
        "name": "run_python",
        "description": "Execute Python code in a sandbox",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to run"},
            },
            "required": ["code"],
        },
    },
}


async def execute_mock_tool(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Execute a mock tool and return simulated results."""
    if tool_name == "web_search":
        query = params.get("query", "")
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Mock search results for: {query}\n"
                    f"1. Result 1 - A relevant article about {query}\n"
                    f"2. Result 2 - Another resource on {query}\n"
                    f"3. Result 3 - Documentation related to {query}",
                }
            ],
        }

    elif tool_name == "rag_search":
        query = params.get("query", "")
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Mock RAG results for: {query}\n"
                    f"Found 3 relevant documents in knowledge base.\n"
                    f"Document 1: Internal guide about {query}\n"
                    f"Document 2: FAQ section mentioning {query}",
                }
            ],
        }

    elif tool_name == "run_python":
        code = params.get("code", "")
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Mock execution result:\n"
                    f"Code: {code[:100]}...\n"
                    f"Output: [Simulated output]",
                }
            ],
        }

    return {
        "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
        "isError": True,
    }
