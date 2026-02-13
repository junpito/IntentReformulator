"""
Intent Reformulator Module
Core logic for extracting intent proposals from chat history via LLM.
The proposal is Non-Binding — it must be confirmed by the user before
being committed to the Deterministic Kernel (V1).
"""

from __future__ import annotations

import logging
from typing import Any

from src.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)

# Official intent list — the only verbs accepted as valid proposals
ALLOWED_INTENTS: list[str] = [
    "refund",
    "exchange",
    "return",
    "order_status",
    "cancel_order",
    "complaint",
]


class IntentReformulator:
    """
    Extracts and validates intent proposals from multi-turn chat history.

    The reformulator calls the LLM to extract a verb (intent) and object
    from the conversation, then validates the verb against the official
    allowed intents list.
    """

    def __init__(self, llm_client: BaseLLMClient, system_prompt: str) -> None:
        """
        Initialize the IntentReformulator.

        Args:
            llm_client: An LLM client instance (GeminiClient or OpenAIClient).
            system_prompt: The system prompt loaded from YAML.
        """
        self.llm = llm_client
        self.system_prompt = system_prompt

    def process_chat(self, chat_history: list[dict[str, str]]) -> dict[str, Any]:
        """
        Process chat history and return an intent proposal.

        Args:
            chat_history: List of {"role": "user"/"assistant", "content": "..."}.

        Returns:
            A dict with:
              - status: "PROPOSED" if a valid intent was extracted,
                        "AMBIGUOUS" if the intent is unclear or not recognized.
              - intent: The extracted intent verb, or None if ambiguous.
              - target: The extracted object/target, or None.
        """
        if not chat_history:
            logger.warning("Empty chat history provided.")
            return {"status": "AMBIGUOUS", "intent": None, "target": None}

        raw_output = self.llm.generate_structured_output(
            system_prompt=self.system_prompt,
            chat_history=chat_history,
        )

        verb = raw_output.verb.lower().strip()
        target = raw_output.object

        logger.info("Raw LLM extraction — verb: %s, object: %s", verb, target)

        if verb in ALLOWED_INTENTS:
            logger.info("Intent PROPOSED: %s for target: %s", verb, target)
            return {
                "status": "PROPOSED",
                "intent": verb,
                "target": target,
            }

        logger.info("Intent AMBIGUOUS: verb '%s' not in allowed list.", verb)
        return {"status": "AMBIGUOUS", "intent": None, "target": None}
