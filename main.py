"""
LLM Intent Reformulator — Interactive CLI Entry Point

This is the main entry point for testing the Non-Binding Intent Reformulator.
It provides an interactive CLI where you can chat with the system and see
how intents are extracted and proposed for confirmation.
"""

from __future__ import annotations

import logging
import os
import sys

import yaml
from dotenv import load_dotenv

from src.llm_client import GeminiClient
from src.memory import ChatMemory
from src.reformulator import IntentReformulator

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_system_prompt(path: str = "prompts/intent_extraction.yaml") -> str:
    """Load the system prompt from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["system_prompt"]


def print_banner() -> None:
    """Print a welcome banner."""
    print("\n" + "=" * 60)
    print("  LLM Intent Reformulator — Interactive CLI")
    print("  Type your messages to chat. Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")


def format_proposal(proposal: dict) -> str:
    """Format a proposal dict into a readable confirmation message."""
    intent = proposal["intent"]
    target = proposal["target"]

    intent_display = intent.replace("_", " ").title()

    if target:
        return f'I understand you want to perform "{intent_display}" for "{target}". Is this correct?'
    return f'I understand you want to perform "{intent_display}". Is this correct?'


def main() -> None:
    """Main interactive loop."""
    # Load environment variables
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("\n[ERROR] GEMINI_API_KEY not found or not configured.")
        print("Please copy .env.example to .env and add your Gemini API key.")
        print("Get your key at: https://aistudio.google.com/app/apikey\n")
        sys.exit(1)

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Initialize components
    system_prompt = load_system_prompt()
    llm_client = GeminiClient(api_key=api_key, model=model)
    memory = ChatMemory(max_turns=10)
    reformulator = IntentReformulator(llm_client=llm_client, system_prompt=system_prompt)

    print_banner()

    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("\nGoodbye!")
            break

        # Add user message to memory
        memory.add_message("user", user_input)

        # Process through reformulator
        try:
            proposal = reformulator.process_chat(memory.get_history())
        except Exception as e:
            logger.error("Error processing chat: %s", e)
            print(f"\n[ERROR] Failed to process: {e}")
            print("Please try again.\n")
            continue

        # Handle the proposal
        if proposal["status"] == "PROPOSED":
            print(f"\n[PROPOSAL] {format_proposal(proposal)}")
            print("  [1] Yes, proceed")
            print("  [2] No, that's not what I meant\n")

            try:
                choice = input("Your choice (1/2): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break

            if choice == "1":
                # User confirmed — create J01 Commit Boundary Event
                print("\n" + "-" * 50)
                print("[CONFIRMED] J01 Commit Boundary Event:")
                print(f"  Intent : {proposal['intent']}")
                print(f"  Target : {proposal['target']}")
                print(f"  Status : COMMITTED")
                print("-" * 50)
                print(">> This data is now ready to be sent to V1 Deterministic Kernel.")
                print()

                # Reset memory for next conversation
                memory.clear()

                # Add a friendly assistant message
                bot_response = f"Done! Your {proposal['intent'].replace('_', ' ')} request has been submitted. How else can I help you?"
                print(f"Bot: {bot_response}\n")

            else:
                # User declined — reset and ask again
                memory.clear()
                bot_response = "No problem! Let's start over. Please describe what you need help with."
                print(f"\nBot: {bot_response}\n")

        elif proposal["status"] == "AMBIGUOUS":
            # Intent unclear — ask for clarification
            bot_response = "I'm not sure I understand your request. Could you please describe more specifically what you'd like to do? For example: refund, exchange, return, check order status, cancel order, or file a complaint."
            memory.add_message("assistant", bot_response)
            print(f"\nBot: {bot_response}\n")


if __name__ == "__main__":
    main()
