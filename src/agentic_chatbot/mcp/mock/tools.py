"""Mock tool implementations for testing."""

from abc import ABC, abstractmethod
from typing import Any

from agentic_chatbot.mcp.models import ToolContent, ToolResult, ToolResultStatus


class MockTool(ABC):
    """Base class for mock tools."""

    name: str
    description: str
    input_schema: dict[str, Any]

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute the mock tool."""
        ...


class MockRAGSearchTool(MockTool):
    """Mock RAG search tool."""

    name = "rag_search"
    description = "Search internal knowledge base"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "default": 5, "description": "Number of results"},
        },
        "required": ["query"],
    }

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        query = params.get("query", "")
        top_k = params.get("top_k", 5)

        # Mock results
        results = [
            {
                "title": f"Document {i}",
                "content": f"This is mock content for '{query}' - result {i}",
                "score": 0.9 - (i * 0.1),
            }
            for i in range(min(top_k, 3))
        ]

        return ToolResult.success(
            tool_name=self.name,
            contents=[
                ToolContent(
                    content_type="application/json",
                    data={"results": results, "total": len(results)},
                )
            ],
        )


class MockWebSearchTool(MockTool):
    """Mock web search tool."""

    name = "web_search"
    description = "Search the web for information"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "num_results": {"type": "integer", "default": 10, "description": "Number of results"},
        },
        "required": ["query"],
    }

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        query = params.get("query", "")
        num_results = params.get("num_results", 10)

        # Mock web results
        results = [
            {
                "title": f"Web Result {i}: {query}",
                "url": f"https://example.com/result{i}",
                "snippet": f"Mock snippet about '{query}' from the web...",
            }
            for i in range(min(num_results, 5))
        ]

        return ToolResult.success(
            tool_name=self.name,
            contents=[
                ToolContent.markdown(
                    "\n\n".join(
                        f"**{r['title']}**\n{r['url']}\n{r['snippet']}" for r in results
                    )
                )
            ],
        )


class MockCodeExecutorTool(MockTool):
    """Mock code executor tool."""

    name = "run_python"
    description = "Execute Python code in a sandbox"
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
        },
        "required": ["code"],
    }

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        code = params.get("code", "")

        # Mock execution (just return the code with a mock output)
        mock_output = f"# Code executed successfully\n# Input:\n{code}\n\n# Output:\nMock execution result"

        return ToolResult.success(
            tool_name=self.name,
            contents=[
                ToolContent.text(mock_output),
            ],
            metadata={"execution_time_ms": 100},
        )


class MockDataAnalyzerTool(MockTool):
    """Mock data analyzer tool that returns multi-modal content."""

    name = "analyze_data"
    description = "Analyze data and return insights with visualizations"
    input_schema = {
        "type": "object",
        "properties": {
            "data": {"type": "string", "description": "Data to analyze"},
            "analysis_type": {
                "type": "string",
                "enum": ["summary", "trend", "comparison"],
                "default": "summary",
            },
        },
        "required": ["data"],
    }

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        data = params.get("data", "")
        analysis_type = params.get("analysis_type", "summary")

        # Mock multi-modal response with text and "chart"
        text_content = ToolContent.markdown(
            f"## {analysis_type.title()} Analysis\n\n"
            f"Analyzed data: {data[:100]}...\n\n"
            "### Key Findings\n"
            "- Finding 1: Mock insight about the data\n"
            "- Finding 2: Another mock observation\n"
            "- Finding 3: Third mock discovery\n"
        )

        # Mock "chart" as a widget
        widget_content = ToolContent.widget(
            {
                "widget_type": "chart",
                "title": f"{analysis_type.title()} Chart",
                "data": {
                    "type": "bar",
                    "labels": ["A", "B", "C", "D"],
                    "values": [10, 20, 15, 25],
                },
            }
        )

        return ToolResult.success(
            tool_name=self.name,
            contents=[text_content, widget_content],
        )


def create_default_mock_tools() -> list[MockTool]:
    """Create the default set of mock tools."""
    return [
        MockRAGSearchTool(),
        MockWebSearchTool(),
        MockCodeExecutorTool(),
        MockDataAnalyzerTool(),
    ]
