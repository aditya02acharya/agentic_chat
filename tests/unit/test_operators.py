"""Tests for operator system."""

import pytest

from agentic_chatbot.operators.base import BaseOperator, OperatorType
from agentic_chatbot.operators.registry import OperatorRegistry
from agentic_chatbot.operators.context import OperatorContext, OperatorResult


class TestOperatorRegistry:
    """Tests for operator registry."""

    def setup_method(self):
        """Clear registry before each test."""
        OperatorRegistry.clear()

    def test_register_operator(self):
        """Test registering an operator."""

        @OperatorRegistry.register("test_op")
        class TestOperator(BaseOperator):
            name = "test_op"
            description = "Test operator"
            operator_type = OperatorType.PURE_LLM

            async def execute(self, context, mcp_session=None):
                return OperatorResult.success_result("test")

        assert OperatorRegistry.exists("test_op")
        assert "test_op" in OperatorRegistry.list_operators()

    def test_create_operator(self):
        """Test creating operator instance."""

        @OperatorRegistry.register("test_op_2")
        class TestOperator(BaseOperator):
            name = "test_op_2"
            description = "Test operator 2"
            operator_type = OperatorType.PURE_LLM

            async def execute(self, context, mcp_session=None):
                return OperatorResult.success_result("test")

        op = OperatorRegistry.create("test_op_2")
        assert op.name == "test_op_2"
        assert op.operator_type == OperatorType.PURE_LLM

    def test_unknown_operator(self):
        """Test creating unknown operator raises error."""
        with pytest.raises(KeyError):
            OperatorRegistry.create("nonexistent")


class TestOperatorContext:
    """Tests for operator context."""

    def test_basic_context(self):
        """Test basic context creation."""
        ctx = OperatorContext(query="test query")
        assert ctx.query == "test query"
        assert ctx.recent_messages == []

    def test_extra_context(self):
        """Test extra context storage."""
        ctx = OperatorContext(query="test")
        ctx.set("custom_key", "custom_value")
        assert ctx.get("custom_key") == "custom_value"
        assert ctx.get("missing", "default") == "default"


class TestOperatorResult:
    """Tests for operator result."""

    def test_success_result(self):
        """Test creating success result."""
        result = OperatorResult.success_result("output data")
        assert result.success
        assert result.output == "output data"
        assert result.error is None

    def test_error_result(self):
        """Test creating error result."""
        result = OperatorResult.error_result("Something went wrong")
        assert not result.success
        assert result.error == "Something went wrong"

    def test_text_output(self):
        """Test text output property."""
        result = OperatorResult.success_result("plain text")
        assert result.text_output == "plain text"

        result2 = OperatorResult.success_result({"key": "value"})
        assert "key" in result2.text_output
