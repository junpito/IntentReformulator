# Intent Reformulator - Full Documentation

Complete technical documentation for every feature, class, function, and concept
in this project. Designed to help you understand the codebase from top to bottom.

> **Note:** A multi-provider version (Gemini + OpenAI) is available on the
> [`feature/multi-provider`](https://github.com/junpito/IntentReformulator/tree/feature/multi-provider) branch.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Data Flow](#2-architecture--data-flow)
3. [File-by-File Breakdown](#3-file-by-file-breakdown)
   - 3.1 [prompts/intent_extraction.yaml](#31-promptsintent_extractionyaml)
   - 3.2 [src/memory.py — ChatMemory](#32-srcmemorypy--chatmemory)
   - 3.3 [src/llm_client.py — OpenAIClient](#33-srcllm_clientpy--openaiclient)
   - 3.4 [src/reformulator.py — IntentReformulator](#34-srcreformulatorpy--intentreformulator)
   - 3.5 [main.py — CLI Entry Point](#35-mainpy--cli-entry-point)
   - 3.6 [tests/test_multiturn.py — Test Suite](#36-teststest_multiturnpy--test-suite)
4. [Key Concepts Explained](#4-key-concepts-explained)
5. [Complete Data Flow Walkthrough](#5-complete-data-flow-walkthrough)
6. [How to Add New Intents](#6-how-to-add-new-intents)

---

## 1. Project Overview

This project is a **Non-Binding Intent Reformulator**. Its job:

1. Read a multi-turn chat conversation between a customer and a support bot
2. Use **OpenAI ChatGPT** to extract **what the customer wants** (intent + object)
3. **Propose** the extracted intent to the user for confirmation
4. Only after user confirms, the intent is "committed" (ready to be sent to V1 Kernel)

**"Non-Binding"** means the LLM's output is never trusted blindly. It is always a
*proposal* that requires human confirmation. This protects against LLM hallucination.

---

## 2. Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py (CLI)                           │
│  Handles: user input, display, confirmation flow                │
└──────┬──────────────────────────────────┬───────────────────────┘
       │                                  │
       ▼                                  ▼
┌──────────────┐                 ┌────────────────────┐
│  ChatMemory  │                 │ IntentReformulator  │
│  (memory.py) │ ── history ──▶ │  (reformulator.py)  │
│              │                 │                     │
│ Stores chat  │                 │ Validates intent    │
│ messages in  │                 │ against allowed     │
│ sliding      │                 │ list                │
│ window       │                 └─────────┬──────────┘
└──────────────┘                           │
                                           ▼
                                 ┌────────────────────┐
                                 │   OpenAIClient      │
                                 │   (llm_client.py)   │
                                 │                     │
                                 │ Sends prompt +      │
                                 │ history to ChatGPT  │
                                 │ API, returns        │
                                 │ structured JSON     │
                                 └─────────┬──────────┘
                                           │
                                           ▼
                                 ┌────────────────────┐
                                 │   OpenAI API        │
                                 │  (External Service) │
                                 └────────────────────┘
```

**Data flows in one direction:**
User Input → Memory → Reformulator → LLM Client → OpenAI API → back up → User Confirmation

---

## 3. File-by-File Breakdown

---

### 3.1 `prompts/intent_extraction.yaml`

**Purpose:** Stores the System Prompt — the instruction that tells the LLM *how* to behave.

**Format:** YAML with a single key `system_prompt` containing a multi-line string.

**What the prompt tells the LLM:**

| Rule # | What It Does |
|--------|-------------|
| 1 | Use the ENTIRE conversation to resolve pronouns ("it", "that one") |
| 2 | The verb MUST come from the 6 official intents only |
| 3 | The object should be the specific item/order mentioned |
| 4 | If unclear, output verb = "unknown", object = null |
| 5 | Focus on the customer's MOST RECENT message |
| 6 | Do NOT invent information |

**Why YAML?** So you can change the prompt without editing Python code. This is
a common pattern called "prompt-as-config" — it makes prompt iteration fast.

**How it's loaded:**
```python
# In main.py
with open("prompts/intent_extraction.yaml", "r") as f:
    data = yaml.safe_load(f)
system_prompt = data["system_prompt"]  # Returns the prompt as a plain string
```

---

### 3.2 `src/memory.py` — ChatMemory

**Purpose:** Manages the multi-turn conversation history with a sliding window.

#### Class: `ChatMemory`

```python
class ChatMemory:
    def __init__(self, max_turns: int = 10)
```

**Constructor Parameter:**
- `max_turns` (default: 10) — Maximum number of conversation *turns* to keep.
  One turn = one user message + one assistant reply = 2 messages.
  So `max_turns=10` keeps up to 20 messages.

**Internal Storage:**
```python
self._history: list[dict[str, str]] = []
# Example content:
# [
#     {"role": "user", "content": "Check order #001"},
#     {"role": "assistant", "content": "Order #001 is being shipped."},
#     {"role": "user", "content": "Cancel it."},
# ]
```

#### Methods:

| Method | Signature | What It Does |
|--------|-----------|-------------|
| `add_message` | `(role: str, content: str) → None` | Adds a message to history. `role` must be `"user"` or `"assistant"`. After adding, it automatically calls `_trim()` to enforce the sliding window. Raises `ValueError` if role is invalid. |
| `get_history` | `() → list[dict]` | Returns a **copy** of the history (not a reference). This prevents external code from accidentally modifying the internal state. |
| `clear` | `() → None` | Empties all history. Called when user declines a proposal or after a successful commit. |
| `_trim` | `() → None` | **Private method.** Enforces the sliding window by removing the oldest messages when history exceeds `max_turns * 2`. |
| `__len__` | `() → int` | Lets you do `len(memory)` to get the number of messages. |
| `__repr__` | `() → str` | Lets you do `print(memory)` to see a summary like `ChatMemory(max_turns=10, messages=5)`. |

#### Sliding Window Explained:

```
max_turns = 2 (keeps 4 messages)

Add msg1, resp1, msg2, resp2, msg3, resp3:

Before trim: [msg1, resp1, msg2, resp2, msg3, resp3]  (6 messages)
After trim:  [msg2, resp2, msg3, resp3]                (4 messages)

The oldest messages (msg1, resp1) are removed.
```

**Why sliding window?** LLMs have a context limit (max tokens). If you send too
many messages, the API will reject the request or the quality degrades. The window
keeps only the most recent, relevant messages.

---

### 3.3 `src/llm_client.py` — OpenAIClient

**Purpose:** Wraps the OpenAI ChatGPT API. Sends chat history + prompt, gets back structured JSON.

#### Model: `IntentOutput`

```python
class IntentOutput(BaseModel):
    verb: str
    object: str | None = None
```

A **Pydantic model** that defines the expected LLM output shape:
- `verb` — The extracted intent (e.g., `"refund"`, `"cancel_order"`, `"unknown"`)
- `object` — The target item/order (e.g., `"#001"`, `"red shoes"`) or `None`

**Why Pydantic?**
1. **Validation** — Pydantic automatically checks that `verb` is a string and `object`
   is either a string or null. If the LLM returns garbage, parsing will fail cleanly.
2. **Schema enforcement** — OpenAI's `response_format={"type": "json_object"}` forces
   JSON output. Pydantic then validates the structure matches `IntentOutput`.

#### Class: `OpenAIClient`

```python
class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    )
```

**Constructor Parameters:**
| Parameter | Default | What It Does |
|-----------|---------|-------------|
| `api_key` | (required) | Your OpenAI API key |
| `model` | `"gpt-4o-mini"` | Which ChatGPT model to use. gpt-4o-mini is fast and cheap. |
| `max_retries` | `3` | How many times to retry if the API call fails |
| `retry_delay` | `1.0` | Base delay (seconds) between retries. Uses exponential backoff: 1s, 2s, 4s... |

**Initialization:**
```python
self.client = OpenAI(api_key=api_key)
# Creates the official OpenAI SDK client
```

#### Method: `generate_structured_output(system_prompt, chat_history) → IntentOutput`

This is the main method. It:

1. **Builds messages** — prepends `{"role": "system", "content": system_prompt}` then appends chat history
2. **Calls** the OpenAI API with:
   - `messages` — System prompt + chat history
   - `response_format={"type": "json_object"}` — Forces JSON output
   - `temperature=0` — Makes output deterministic (same input = same output)
3. **Parses** the response with `IntentOutput.model_validate_json()`
4. **Retries** on failure with exponential backoff

**How messages are built:**
```python
messages = [
    {"role": "system", "content": "You are a data extraction assistant..."},
    {"role": "user", "content": "Check order #001"},
    {"role": "assistant", "content": "Order #001 is being shipped."},
    {"role": "user", "content": "Cancel it."},
]
```

OpenAI uses `"assistant"` for bot messages — the same format as our internal
`ChatMemory`, so no role translation is needed.

**Retry Logic Explained:**
```
Attempt 1: Call API
  → If fails: wait 1 second
Attempt 2: Call API
  → If fails: wait 2 seconds (1.0 * 2^1)
Attempt 3: Call API
  → If fails: raise RuntimeError (all retries exhausted)
```

This handles transient errors like network timeouts or rate limiting.

---

### 3.4 `src/reformulator.py` — IntentReformulator

**Purpose:** The core brain — takes chat history, calls LLM, validates the result.

#### Constant: `ALLOWED_INTENTS`

```python
ALLOWED_INTENTS = [
    "refund",       # Customer wants money back
    "exchange",     # Customer wants to swap product
    "return",       # Customer wants to send product back
    "order_status", # Customer wants to check order
    "cancel_order", # Customer wants to cancel order
    "complaint",    # Customer has a complaint
]
```

This is the **official intent list**. Only these 6 verbs are accepted as valid proposals.
Anything else (including LLM hallucinations) is rejected as AMBIGUOUS.

#### Class: `IntentReformulator`

```python
class IntentReformulator:
    def __init__(self, llm_client: OpenAIClient, system_prompt: str)
```

**Constructor Parameters:**
- `llm_client` — An instance of `OpenAIClient`
- `system_prompt` — The prompt string loaded from YAML

##### `process_chat(chat_history) → dict`

This is the **main entry point** for intent extraction. It:

1. **Guards** against empty history (returns AMBIGUOUS immediately)
2. **Calls** `llm_client.generate_structured_output()` with the prompt + history
3. **Normalizes** the verb: lowercase + strip whitespace
4. **Validates** the verb against `ALLOWED_INTENTS`
5. **Returns** a proposal dict

**Return value — two possible shapes:**

```python
# If intent is recognized:
{
    "status": "PROPOSED",
    "intent": "cancel_order",    # One of the 6 allowed intents
    "target": "#001"             # The extracted object, or None
}

# If intent is unclear or not recognized:
{
    "status": "AMBIGUOUS",
    "intent": None,
    "target": None
}
```

**Why this is "Non-Binding":** This method returns a *proposal*, not a *command*.
The calling code (main.py) must ask the user to confirm before acting on it.

---

### 3.5 `main.py` — CLI Entry Point

**Purpose:** Interactive CLI for testing. Ties all components together.

#### Functions:

##### `load_system_prompt(path) → str`

Opens the YAML file, parses it with `yaml.safe_load()`, and returns the
`system_prompt` value as a plain string.

##### `print_banner(model) → None`

Prints a welcome message showing the current model name.

##### `format_proposal(proposal) → str`

Takes a proposal dict and formats it into a human-readable confirmation question:

```
Input:  {"status": "PROPOSED", "intent": "cancel_order", "target": "#001"}
Output: 'I understand you want to perform "Cancel Order" for "#001". Is this correct?'
```

##### `main() → None`

The main loop:

```
1. Load .env → get OPENAI_API_KEY + OPENAI_MODEL
2. Validate API key exists
3. Initialize: OpenAIClient → ChatMemory → IntentReformulator
4. LOOP:
   a. Get user input
   b. If "quit"/"exit" → break
   c. Add message to memory
   d. Call reformulator.process_chat(history)
   e. If PROPOSED:
      - Show proposal to user
      - Ask: [1] Yes  [2] No
      - If Yes → print J01 Commit Event, clear memory
      - If No  → clear memory, start over
   f. If AMBIGUOUS:
      - Ask user to clarify
      - Add assistant message to memory (so context is preserved)
```

---

### 3.6 `tests/test_multiturn.py` — Test Suite

**Purpose:** Ensures everything works correctly. Two types of tests:

#### Unit Tests (no API key needed)

These use `unittest.mock.MagicMock` to simulate the LLM.

| Test | What It Verifies |
|------|-----------------|
| `test_add_and_retrieve_messages` | ChatMemory stores and returns messages correctly |
| `test_sliding_window_trims_old_messages` | Old messages are removed when window is exceeded |
| `test_clear_empties_history` | `clear()` resets everything to empty |
| `test_invalid_role_raises_error` | Using role `"system"` raises `ValueError` |
| `test_get_history_returns_copy` | Modifying returned history doesn't affect internal state |
| `test_multiturn_cancel_order` | **Golden Test:** Multi-turn chat correctly extracts `cancel_order` + `#001` |
| `test_ambiguous_input` | **Golden Test:** Unclear input returns `AMBIGUOUS` |
| `test_refund_proposal` | Single-turn refund extraction works |
| `test_return_proposal` | Single-turn return extraction works |
| `test_empty_history_returns_ambiguous` | Empty history returns AMBIGUOUS without calling LLM |
| `test_all_allowed_intents_are_accepted` | All 6 intents produce PROPOSED status |

#### Integration Tests (API key required)

These call the **real OpenAI API** to verify end-to-end behavior.

| Test | What It Verifies |
|------|-----------------|
| `test_multiturn_cancel_order_integration` | Real LLM resolves "cancel it" → cancel_order for #001 |
| `test_ambiguous_integration` | Real LLM returns unknown for "I'm confused." |
| `test_refund_request_integration` | Real LLM extracts refund intent correctly |

These tests are **skipped automatically** if `OPENAI_API_KEY` is not set in `.env`.

**Run commands:**
```bash
pytest tests/ -v                    # Unit tests only
pytest tests/ -v -m integration     # Integration tests only
```

---

## 4. Key Concepts Explained

### 4.1 Non-Binding Proposal

The LLM's output is **never executed directly**. It's always a "proposal" that
the user must confirm:

```
LLM says: cancel_order for #001
  ↓
System asks: "Do you want to cancel order #001?"
  ↓
User says YES → Only THEN is it committed
User says NO  → Proposal is discarded, no harm done
```

This prevents LLM hallucinations from causing real actions.

### 4.2 J01 Commit Boundary Event

The formal "handoff point" between the LLM module (this project)
and the V1 Deterministic Kernel. Currently it's just a printed message,
but in production it would be a structured event sent via HTTP/queue.

### 4.3 Epistemic Boundary

This project is the **boundary between uncertain knowledge (LLM) and certain action (V1 Kernel)**.

- **Before the boundary:** LLM guesses, proposes, might be wrong
- **At the boundary:** User confirms (Yes/No)
- **After the boundary:** Deterministic code executes with 100% certainty

### 4.4 Structured Output (Pydantic + OpenAI)

Instead of asking the LLM to output free text and then parsing it with regex,
we use OpenAI's JSON mode:

```python
response = client.chat.completions.create(
    response_format={"type": "json_object"},  # Force JSON output
    temperature=0,
)
# Then validate with Pydantic:
IntentOutput.model_validate_json(response.choices[0].message.content)
```

This ensures the output is valid JSON, and Pydantic validates the structure.

### 4.5 Temperature = 0

```python
temperature=0
```

Temperature controls randomness in LLM output:
- `temperature=0` → Always picks the most likely answer (deterministic)
- `temperature=1` → More creative/random

For intent extraction, we want consistency. That's why we use `temperature=0`.

### 4.6 Sliding Window

LLMs have a maximum context size. The sliding window keeps the N most recent
messages and discards older ones:

```
Window size = 4 messages (2 turns)

Full history:  [A, B, C, D, E, F]
After window:  [      C, D, E, F]  ← Only last 4 kept
```

---

## 5. Complete Data Flow Walkthrough

**Scenario:** User asks about order #001, then wants to cancel it.

### Step 1: User types "Check status of order #001"

```python
memory.add_message("user", "Check status of order #001")
```

### Step 2: Reformulator processes

```python
proposal = reformulator.process_chat(memory.get_history())

# Inside reformulator → calls llm_client.generate_structured_output()
# Inside OpenAIClient:
#   1. Builds messages: [system_prompt, user_msg]
#   2. Sends to OpenAI with temperature=0, response_format=json_object
#   3. ChatGPT returns: {"verb": "order_status", "object": "#001"}
#   4. Parsed into: IntentOutput(verb="order_status", object="#001")
# Back in reformulator:
#   verb = "order_status" → in ALLOWED_INTENTS? YES
#   Returns: {"status": "PROPOSED", "intent": "order_status", "target": "#001"}
```

### Step 3: Bot replies, user says "Actually, cancel it."

```python
memory.add_message("assistant", "Order #001 is being shipped.")
memory.add_message("user", "Actually, cancel it.")
```

### Step 4: LLM resolves "it" from context

The LLM sees the full history and knows "it" = order #001. It outputs:
```json
{"verb": "cancel_order", "object": "#001"}
```

### Step 5: User confirms → J01 Event

```
[CONFIRMED] J01 Commit Boundary Event:
  Intent : cancel_order
  Target : #001
  Status : COMMITTED
```

Memory is cleared. Ready for next conversation.

---

## 6. How to Add New Intents

### Step 1: Update `ALLOWED_INTENTS` in `src/reformulator.py`

```python
ALLOWED_INTENTS: list[str] = [
    "refund",
    "exchange",
    "return",
    "order_status",
    "cancel_order",
    "complaint",
    "track_delivery",  # ← Add here
]
```

### Step 2: Update the system prompt in `prompts/intent_extraction.yaml`

```yaml
  Official Intent List:
    - refund
    - exchange
    - return
    - order_status
    - cancel_order
    - complaint
    - track_delivery    # ← Add here
```

### Step 3: Add a test case in `tests/test_multiturn.py`

```python
def test_track_delivery_proposal(self):
    reformulator = self._make_reformulator("track_delivery", "#002")
    chat_history = [
        {"role": "user", "content": "Where is my package #002?"},
    ]
    result = reformulator.process_chat(chat_history)
    assert result["status"] == "PROPOSED"
    assert result["intent"] == "track_delivery"
```

**Important:** You must update BOTH `ALLOWED_INTENTS` (code) AND the YAML prompt
(LLM instruction). If you only update one, the system won't work correctly.
