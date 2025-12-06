"""Tests for context management."""

import pytest

from agentic_chatbot.context.memory import ConversationMemory, Message
from agentic_chatbot.context.results import ResultStore
from agentic_chatbot.context.actions import ActionHistory
from agentic_chatbot.operators.context import OperatorResult


class TestConversationMemory:
    """Tests for conversation memory."""

    def test_add_message(self):
        """Test adding messages."""
        memory = ConversationMemory(window_size=5)
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi there!")

        assert len(memory) == 2

    def test_get_recent(self):
        """Test getting recent messages."""
        memory = ConversationMemory(window_size=3)

        for i in range(5):
            memory.add_message("user", f"Message {i}")

        recent = memory.get_recent(3)
        assert len(recent) == 3
        assert recent[0].content == "Message 2"

    @pytest.mark.asyncio
    async def test_get_method(self):
        """Test get method with DSL."""
        memory = ConversationMemory(window_size=5)
        memory.add_message("user", "Test")
        memory.set_summary("Previous conversation summary")

        recent = await memory.get("recent(5)")
        assert len(recent) == 1

        summary = await memory.get("summary")
        assert summary == "Previous conversation summary"

    def test_format_for_prompt(self):
        """Test formatting for prompts."""
        memory = ConversationMemory()
        memory.add_message("user", "Hello")
        memory.set_summary("Earlier discussion")

        formatted = memory.format_for_prompt()
        assert "Earlier discussion" in formatted
        assert "Hello" in formatted


class TestResultStore:
    """Tests for result store."""

    def test_store_result(self):
        """Test storing results."""
        store = ResultStore()
        result = OperatorResult.success_result("test output")

        store.store("key1", result, source="test_tool")

        assert len(store) == 1
        assert store.has_results

    def test_get_step_result(self):
        """Test getting step results."""
        store = ResultStore()
        result = OperatorResult.success_result("step output")

        store.store_step_result("step_1", result)

        retrieved = store.get_step_result("step_1")
        assert retrieved is not None
        assert retrieved.output == "step output"

    @pytest.mark.asyncio
    async def test_get_method(self):
        """Test get method with DSL."""
        store = ResultStore()
        result = OperatorResult.success_result("output")
        store.store_step_result("step_2", result)

        step_result = await store.get("step(step_2)")
        assert step_result is not None

        all_results = await store.get("all")
        assert "step_step_2" in all_results


class TestActionHistory:
    """Tests for action history."""

    def test_record_action(self):
        """Test recording actions."""
        history = ActionHistory()
        history.record("CALL_TOOL", "Called web search", success=True)

        assert len(history) == 1
        actions = history.get_this_turn()
        assert "CALL_TOOL" in actions[0]

    def test_failed_action(self):
        """Test recording failed actions."""
        history = ActionHistory()
        history.record("CALL_TOOL", "Failed search", success=False, error="Timeout")

        failed = history.get_failed()
        assert len(failed) == 1
        assert failed[0].error == "Timeout"

    def test_has_action_type(self):
        """Test checking for action types."""
        history = ActionHistory()
        history.record("CALL_TOOL", "test")

        assert history.has_action_type("CALL_TOOL")
        assert not history.has_action_type("CREATE_WORKFLOW")

    def test_start_new_turn(self):
        """Test starting a new turn."""
        history = ActionHistory()
        history.record("CALL_TOOL", "test")
        history.start_new_turn()

        assert len(history) == 0
