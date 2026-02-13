"""
LLM Client Module
Wrapper for Google Gemini API using the official google-genai SDK.
Provides structured JSON output via Pydantic schema enforcement.
"""

from __future__ import annotations

import logging
import time

from google import genai
from google.genai import types
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class IntentOutput(BaseModel):
    """Pydantic model for structured LLM output."""
    verb: str
    object: str | None = None


class GeminiClient:
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
        # Build the contents list for multi-turn conversation
        contents = self._build_contents(chat_history)

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug("API call attempt %d/%d", attempt, self.max_retries)

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
                logger.info("LLM output: verb=%s, object=%s", result.verb, result.object)
                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    "Attempt %d failed: %s. %s",
                    attempt,
                    type(e).__name__,
                    str(e),
                )
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    logger.debug("Retrying in %.1f seconds...", delay)
                    time.sleep(delay)

        raise RuntimeError(
            f"Failed after {self.max_retries} attempts. Last error: {last_error}"
        )

    @staticmethod
    def _build_contents(
        chat_history: list[dict[str, str]],
    ) -> list[types.Content]:
        """
        Convert chat history dicts to Gemini Content objects.

        Args:
            chat_history: List of {"role": "user"/"assistant", "content": "..."}.

        Returns:
            List of google.genai.types.Content objects.
        """
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
