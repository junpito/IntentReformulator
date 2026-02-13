"""
LLM Client Module
Wrapper for OpenAI ChatGPT API using the official openai SDK.
Provides structured JSON output via Pydantic schema enforcement.
"""

from __future__ import annotations

import logging
import time

from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class IntentOutput(BaseModel):
    """Pydantic model for structured LLM output."""
    verb: str
    object: str | None = None


class OpenAIClient:
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
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        messages.extend(chat_history)

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug("API call attempt %d/%d", attempt, self.max_retries)

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0,
                )

                raw_text = response.choices[0].message.content
                result = IntentOutput.model_validate_json(raw_text)
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
