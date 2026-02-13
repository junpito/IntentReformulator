"""
Golden Test Cases for LLM Intent Reformulator

Contains both unit tests (with mocked LLM) and integration tests (with real API).
Unit tests run without an API key; integration tests require API keys in .env.

Run unit tests:     pytest tests/ -v
Run all tests:      pytest tests/ -v -m "integration or not integration"
Run integration:    pytest tests/ -v -m integration
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv

from src.llm_client import (
    BaseLLMClient,
    GeminiClient,
    IntentOutput,
    OpenAIClient,
    create_llm_client,
)
from src.memory import ChatMemory
from src.reformulator import ALLOWED_INTENTS, IntentReformulator


# ===========================================================================
# Unit Tests — ChatMemory
# ===========================================================================

class TestChatMemory:
    """Tests for the ChatMemory sliding window."""

    def test_add_and_retrieve_messages(self):
        memory = ChatMemory(max_turns=5)
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi there!")

        history = memory.get_history()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi there!"}

    def test_sliding_window_trims_old_messages(self):
        memory = ChatMemory(max_turns=2)  # keeps last 4 messages (2 turns)

        # Add 3 full turns (6 messages), should keep only last 4
        memory.add_message("user", "msg1")
        memory.add_message("assistant", "resp1")
        memory.add_message("user", "msg2")
        memory.add_message("assistant", "resp2")
        memory.add_message("user", "msg3")
        memory.add_message("assistant", "resp3")

        history = memory.get_history()
        assert len(history) == 4
        assert history[0]["content"] == "msg2"
        assert history[-1]["content"] == "resp3"

    def test_clear_empties_history(self):
        memory = ChatMemory()
        memory.add_message("user", "Hello")
        memory.clear()
        assert len(memory) == 0
        assert memory.get_history() == []

    def test_invalid_role_raises_error(self):
        memory = ChatMemory()
        with pytest.raises(ValueError, match="Invalid role"):
            memory.add_message("system", "This should fail")

    def test_get_history_returns_copy(self):
        memory = ChatMemory()
        memory.add_message("user", "Hello")
        history = memory.get_history()
        history.append({"role": "user", "content": "Injected"})
        assert len(memory.get_history()) == 1  # original unchanged


# ===========================================================================
# Unit Tests — Factory Function
# ===========================================================================

class TestCreateLLMClient:
    """Tests for the create_llm_client factory function."""

    def test_unknown_provider_raises_error(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_client(provider="unsupported", api_key="fake-key")

    def test_provider_name_is_case_insensitive(self):
        """Provider name should be case-insensitive and strip whitespace."""
        # These should not raise ValueError (they will fail at API init,
        # but the provider lookup itself should succeed)
        # We test by checking the error is NOT a ValueError about unknown provider
        for name in ["GEMINI", " Gemini ", "gemini"]:
            try:
                create_llm_client(provider=name, api_key="fake-key")
            except ValueError:
                pytest.fail(f"Provider '{name}' was not recognized")
            except Exception:
                pass  # Other errors (e.g., API init) are fine

        for name in ["OPENAI", " OpenAI ", "openai"]:
            try:
                create_llm_client(provider=name, api_key="fake-key")
            except ValueError:
                pytest.fail(f"Provider '{name}' was not recognized")
            except Exception:
                pass


# ===========================================================================
# Unit Tests — IntentReformulator (with Mocked LLM)
# ===========================================================================

class TestIntentReformulatorMocked:
    """Tests for IntentReformulator with a mocked LLM client."""

    def _make_reformulator(self, verb: str, obj: str | None) -> IntentReformulator:
        """Helper to create a reformulator with a mocked LLM returning fixed output."""
        mock_client = MagicMock(spec=BaseLLMClient)
        mock_client.generate_structured_output.return_value = IntentOutput(
            verb=verb, object=obj
        )
        return IntentReformulator(
            llm_client=mock_client,
            system_prompt="test prompt",
        )

    def test_multiturn_cancel_order(self):
        """
        Golden Test: Multi-turn scenario.
        User asks about order status, then decides to cancel.
        LLM should extract cancel_order for order #001.
        """
        reformulator = self._make_reformulator("cancel_order", "#001")

        chat_history = [
            {"role": "user", "content": "Hi, can I check the status of order #001?"},
            {"role": "assistant", "content": "Your order #001 is currently being shipped."},
            {"role": "user", "content": "Actually, just cancel it."},
        ]

        result = reformulator.process_chat(chat_history)

        assert result["status"] == "PROPOSED"
        assert result["intent"] == "cancel_order"
        assert result["target"] == "#001"

    def test_ambiguous_input(self):
        """
        Golden Test: Ambiguous input.
        User says something unclear, LLM returns 'unknown'.
        """
        reformulator = self._make_reformulator("unknown", None)

        chat_history = [
            {"role": "user", "content": "I'm confused."},
        ]

        result = reformulator.process_chat(chat_history)

        assert result["status"] == "AMBIGUOUS"
        assert result["intent"] is None
        assert result["target"] is None

    def test_refund_proposal(self):
        """User wants a refund for specific item."""
        reformulator = self._make_reformulator("refund", "red shoes")

        chat_history = [
            {"role": "user", "content": "I want a refund for the red shoes I bought."},
        ]

        result = reformulator.process_chat(chat_history)

        assert result["status"] == "PROPOSED"
        assert result["intent"] == "refund"
        assert result["target"] == "red shoes"

    def test_return_proposal(self):
        """User wants to return a product."""
        reformulator = self._make_reformulator("return", "laptop")

        chat_history = [
            {"role": "user", "content": "I'd like to return the laptop I purchased yesterday."},
        ]

        result = reformulator.process_chat(chat_history)

        assert result["status"] == "PROPOSED"
        assert result["intent"] == "return"
        assert result["target"] == "laptop"

    def test_empty_history_returns_ambiguous(self):
        """Empty chat history should return AMBIGUOUS."""
        mock_client = MagicMock(spec=BaseLLMClient)
        reformulator = IntentReformulator(
            llm_client=mock_client, system_prompt="test prompt"
        )

        result = reformulator.process_chat([])

        assert result["status"] == "AMBIGUOUS"
        mock_client.generate_structured_output.assert_not_called()

    def test_all_allowed_intents_are_accepted(self):
        """Every intent in ALLOWED_INTENTS should produce a PROPOSED status."""
        for intent in ALLOWED_INTENTS:
            reformulator = self._make_reformulator(intent, "test_object")
            result = reformulator.process_chat(
                [{"role": "user", "content": "test"}]
            )
            assert result["status"] == "PROPOSED", f"Intent '{intent}' was not accepted"
            assert result["intent"] == intent


# ===========================================================================
# Integration Tests — Requires real API keys
# ===========================================================================

load_dotenv()

HAS_GEMINI_KEY = bool(
    os.getenv("GEMINI_API_KEY")
    and os.getenv("GEMINI_API_KEY") != "your-api-key-here"
)

HAS_OPENAI_KEY = bool(
    os.getenv("OPENAI_API_KEY")
    and os.getenv("OPENAI_API_KEY") != "your-api-key-here"
)


def _load_system_prompt() -> str:
    """Load system prompt from YAML file."""
    import yaml
    with open("prompts/intent_extraction.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["system_prompt"]


# --- Gemini Integration Tests ---

@pytest.mark.integration
@pytest.mark.skipif(not HAS_GEMINI_KEY, reason="GEMINI_API_KEY not set")
class TestIntegrationWithGemini:
    """Integration tests that call the real Gemini API."""

    @pytest.fixture
    def reformulator(self) -> IntentReformulator:
        api_key = os.getenv("GEMINI_API_KEY")
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        client = GeminiClient(api_key=api_key, model=model)
        prompt = _load_system_prompt()
        return IntentReformulator(llm_client=client, system_prompt=prompt)

    def test_multiturn_cancel_order_integration(self, reformulator):
        """
        Integration Golden Test: Multi-turn cancel order.
        Verifies that the real LLM correctly resolves the pronoun 'it'
        to order #001 from the earlier context.
        """
        chat_history = [
            {"role": "user", "content": "Hi, can I check the status of order #001?"},
            {"role": "assistant", "content": "Your order #001 is currently being shipped."},
            {"role": "user", "content": "Actually, just cancel it."},
        ]

        result = reformulator.process_chat(chat_history)

        assert result["status"] == "PROPOSED"
        assert result["intent"] == "cancel_order"
        assert "#001" in (result["target"] or "")

    def test_ambiguous_integration(self, reformulator):
        """
        Integration Golden Test: Ambiguous input.
        Verifies that the real LLM returns 'unknown' for unclear input.
        """
        chat_history = [
            {"role": "user", "content": "I'm confused."},
        ]

        result = reformulator.process_chat(chat_history)

        assert result["status"] == "AMBIGUOUS"
        assert result["intent"] is None

    def test_refund_request_integration(self, reformulator):
        """Integration test: Simple refund request."""
        chat_history = [
            {"role": "user", "content": "I want a refund for my broken headphones."},
        ]

        result = reformulator.process_chat(chat_history)

        assert result["status"] == "PROPOSED"
        assert result["intent"] == "refund"
        assert result["target"] is not None


# --- OpenAI Integration Tests ---

@pytest.mark.integration
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OPENAI_API_KEY not set")
class TestIntegrationWithOpenAI:
    """Integration tests that call the real OpenAI API."""

    @pytest.fixture
    def reformulator(self) -> IntentReformulator:
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        client = OpenAIClient(api_key=api_key, model=model)
        prompt = _load_system_prompt()
        return IntentReformulator(llm_client=client, system_prompt=prompt)

    def test_multiturn_cancel_order_integration(self, reformulator):
        """
        Integration Golden Test: Multi-turn cancel order via OpenAI.
        """
        chat_history = [
            {"role": "user", "content": "Hi, can I check the status of order #001?"},
            {"role": "assistant", "content": "Your order #001 is currently being shipped."},
            {"role": "user", "content": "Actually, just cancel it."},
        ]

        result = reformulator.process_chat(chat_history)

        assert result["status"] == "PROPOSED"
        assert result["intent"] == "cancel_order"
        assert "#001" in (result["target"] or "")

    def test_ambiguous_integration(self, reformulator):
        """
        Integration Golden Test: Ambiguous input via OpenAI.
        """
        chat_history = [
            {"role": "user", "content": "I'm confused."},
        ]

        result = reformulator.process_chat(chat_history)

        assert result["status"] == "AMBIGUOUS"
        assert result["intent"] is None

    def test_refund_request_integration(self, reformulator):
        """Integration test: Simple refund request via OpenAI."""
        chat_history = [
            {"role": "user", "content": "I want a refund for my broken headphones."},
        ]

        result = reformulator.process_chat(chat_history)

        assert result["status"] == "PROPOSED"
        assert result["intent"] == "refund"
        assert result["target"] is not None
