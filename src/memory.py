"""
Chat Memory Module
Manages conversation history with a sliding window to stay within LLM context limits.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ChatMemory:
    """Stores multi-turn chat history with a configurable sliding window."""

    def __init__(self, max_turns: int = 10) -> None:
        """
        Initialize ChatMemory.

        Args:
            max_turns: Maximum number of message pairs to keep.
                       Each user+assistant exchange counts as 2 messages,
                       so max_turns=10 keeps up to 20 messages.
        """
        self.max_turns = max_turns
        self._history: list[dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: Either "user" or "assistant".
            content: The message text.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role '{role}'. Must be 'user' or 'assistant'.")

        self._history.append({"role": role, "content": content})
        self._trim()
        logger.debug("Added %s message. History length: %d", role, len(self._history))

    def get_history(self) -> list[dict[str, str]]:
        """Return a copy of the current conversation history."""
        return list(self._history)

    def clear(self) -> None:
        """Clear all conversation history (used when user declines a proposal)."""
        self._history.clear()
        logger.debug("Chat history cleared.")

    def _trim(self) -> None:
        """Trim history to keep only the last `max_turns * 2` messages."""
        max_messages = self.max_turns * 2
        if len(self._history) > max_messages:
            overflow = len(self._history) - max_messages
            self._history = self._history[overflow:]
            logger.debug("Trimmed %d messages from history.", overflow)

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return f"ChatMemory(max_turns={self.max_turns}, messages={len(self._history)})"
