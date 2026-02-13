"""
LLM Client Module
Provides wrappers for multiple LLM providers (Google Gemini, OpenAI ChatGPT).
Each client returns structured JSON output via Pydantic schema enforcement.

Usage:
    client = create_llm_client(provider="gemini", api_key="...", model="gemini-2.5-flash")
    client = create_llm_client(provider="openai", api_key="...", model="gpt-4o-mini")
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared Pydantic model for structured LLM output
# ---------------------------------------------------------------------------

class IntentOutput(BaseModel):
    """Pydantic model for structured LLM output."""
    verb: str
    object: str | None = None


# ---------------------------------------------------------------------------
# Abstract base class — all LLM clients must implement this interface
# ---------------------------------------------------------------------------

class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.
    Any provider (Gemini, OpenAI, etc.) must implement generate_structured_output().
    """

    @abstractmethod
    def generate_structured_output(
        self,
        system_prompt: str,
        chat_history: list[dict[str, str]],
    ) -> IntentOutput:
        """
        Send chat history to the LLM and get structured intent extraction.

        Args:
            system_prompt: The system instruction for intent extraction.
            chat_history: List of {"role": "user"/"assistant", "content": "..."}.

        Returns:
            IntentOutput with verb and object fields.
        """
        ...


# ---------------------------------------------------------------------------
# Google Gemini Client
# ---------------------------------------------------------------------------

class GeminiClient(BaseLLMClient):
    """Wrapper around the Google Gemini API for intent extraction."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """
        Initialize the Gemini client.

        Args:
            api_key: Google AI Studio API key.
            model: Gemini model name (default: gemini-2.5-flash).
            max_retries: Maximum number of retry attempts on failure.
            retry_delay: Base delay between retries (exponential backoff).
        """
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info("GeminiClient initialized with model: %s", model)

    def generate_structured_output(
        self,
        system_prompt: str,
        chat_history: list[dict[str, str]],
    ) -> IntentOutput:
        """
        Send chat history to Gemini and get structured intent extraction.

        Args:
            system_prompt: The system instruction for intent extraction.
            chat_history: List of {"role": "user"/"assistant", "content": "..."}.

        Returns:
            IntentOutput with verb and object fields.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        from google.genai import types

        contents = self._build_contents(chat_history)

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug("Gemini API call attempt %d/%d", attempt, self.max_retries)

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_schema=IntentOutput,
                        temperature=0,
                    ),
                )

                result = IntentOutput.model_validate_json(response.text)
                logger.info("Gemini output: verb=%s, object=%s", result.verb, result.object)
                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    "Gemini attempt %d failed: %s. %s",
                    attempt,
                    type(e).__name__,
                    str(e),
                )
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    logger.debug("Retrying in %.1f seconds...", delay)
                    time.sleep(delay)

        raise RuntimeError(
            f"Gemini failed after {self.max_retries} attempts. Last error: {last_error}"
        )

    @staticmethod
    def _build_contents(chat_history: list[dict[str, str]]) -> list:
        """Convert chat history dicts to Gemini Content objects."""
        from google.genai import types

        contents = []
        for msg in chat_history:
            # Gemini uses "user" and "model" as role names
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])],
                )
            )
        return contents


# ---------------------------------------------------------------------------
# OpenAI ChatGPT Client
# ---------------------------------------------------------------------------

class OpenAIClient(BaseLLMClient):
    """Wrapper around the OpenAI ChatGPT API for intent extraction."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """
        Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key.
            model: OpenAI model name (default: gpt-4o-mini).
            max_retries: Maximum number of retry attempts on failure.
            retry_delay: Base delay between retries (exponential backoff).
        """
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info("OpenAIClient initialized with model: %s", model)

    def generate_structured_output(
        self,
        system_prompt: str,
        chat_history: list[dict[str, str]],
    ) -> IntentOutput:
        """
        Send chat history to OpenAI and get structured intent extraction.

        Args:
            system_prompt: The system instruction for intent extraction.
            chat_history: List of {"role": "user"/"assistant", "content": "..."}.

        Returns:
            IntentOutput with verb and object fields.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        # Build messages: system prompt + chat history
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(chat_history)

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug("OpenAI API call attempt %d/%d", attempt, self.max_retries)

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0,
                )

                raw_text = response.choices[0].message.content
                result = IntentOutput.model_validate_json(raw_text)
                logger.info("OpenAI output: verb=%s, object=%s", result.verb, result.object)
                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    "OpenAI attempt %d failed: %s. %s",
                    attempt,
                    type(e).__name__,
                    str(e),
                )
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    logger.debug("Retrying in %.1f seconds...", delay)
                    time.sleep(delay)

        raise RuntimeError(
            f"OpenAI failed after {self.max_retries} attempts. Last error: {last_error}"
        )


# ---------------------------------------------------------------------------
# Factory function — create the right client based on provider name
# ---------------------------------------------------------------------------

def create_llm_client(
    provider: str,
    api_key: str,
    model: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> BaseLLMClient:
    """
    Factory function to create an LLM client for the specified provider.

    Args:
        provider: "gemini" or "openai".
        api_key: The API key for the chosen provider.
        model: Model name (optional, uses provider default if not set).
        max_retries: Maximum retry attempts on failure.
        retry_delay: Base delay between retries.

    Returns:
        An instance of BaseLLMClient (GeminiClient or OpenAIClient).

    Raises:
        ValueError: If provider is not recognized.
    """
    provider = provider.lower().strip()

    if provider == "gemini":
        return GeminiClient(
            api_key=api_key,
            model=model or "gemini-2.5-flash",
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
    elif provider == "openai":
        return OpenAIClient(
            api_key=api_key,
            model=model or "gpt-4o-mini",
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
    else:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. Supported: 'gemini', 'openai'."
        )
