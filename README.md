# LLM Intent Reformulator

A **Non-Binding Intent Reformulator** module that uses an LLM to extract user intent from multi-turn chat conversations. Supports **Google Gemini** and **OpenAI ChatGPT** as LLM providers. The extracted intent is proposed to the user for confirmation before being committed to the Deterministic Kernel (V1).

## Architecture

```
User Chat → ChatMemory (sliding window) → IntentReformulator → LLM Client (Gemini / OpenAI)
                                                ↓
                                        Intent Proposal
                                      (PROPOSED / AMBIGUOUS)
                                                ↓
                                      User Confirmation
                                        [Yes] / [No]
                                                ↓
                                   J01 Commit Boundary Event → V1 Kernel
```

## Supported LLM Providers

| Provider | SDK | Default Model | API Key Source |
|---|---|---|---|
| Google Gemini | `google-genai` | `gemini-2.5-flash` | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| OpenAI ChatGPT | `openai` | `gpt-4o-mini` | [OpenAI Platform](https://platform.openai.com/api-keys) |

## Supported Intents

| Intent | Description |
|---|---|
| `refund` | Customer wants a refund |
| `exchange` | Customer wants to exchange a product |
| `return` | Customer wants to return a product |
| `order_status` | Customer wants to check order status |
| `cancel_order` | Customer wants to cancel an order |
| `complaint` | Customer has a complaint |

## Setup

### 1. Get an API Key

**Option A — Google Gemini (free tier available):**
1. Go to **[Google AI Studio](https://aistudio.google.com)** and sign in with your Google account
2. Click **"Get API Key"** in the left sidebar (or go directly to https://aistudio.google.com/app/apikey)
3. Click **"Create API Key"** — Google will automatically create a Cloud project and generate a key
4. Copy the API key

**Option B — OpenAI ChatGPT:**
1. Go to **[OpenAI Platform](https://platform.openai.com/api-keys)** and sign in
2. Click **"Create new secret key"**
3. Copy the API key (you won't be able to see it again)

### 2. Install Dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy the example env file
copy .env.example .env       # Windows
# cp .env.example .env       # Linux/Mac
```

Edit `.env` and configure:
```env
# Choose provider: "gemini" or "openai"
LLM_PROVIDER=gemini

# Set the API key for your chosen provider
GEMINI_API_KEY=your-gemini-key-here
# or
OPENAI_API_KEY=your-openai-key-here
```

### 4. Run

```bash
python main.py
```

### 5. Run Tests

```bash
# Unit tests (no API key needed)
pytest tests/ -v

# Integration tests (requires API key in .env)
pytest tests/ -v -m integration
```

## How It Works

1. **User types a message** — stored in `ChatMemory` with a sliding window (last 10 turns)
2. **IntentReformulator** sends the chat history + system prompt to the LLM
3. **LLM returns structured JSON** — `{"verb": "...", "object": "..."}` via Pydantic schema enforcement
4. **Reformulator validates** — checks if the verb is in the allowed intents list
5. **System proposes** — "I understand you want to [intent] for [object]. Is this correct?"
6. **User confirms or declines**:
   - **Yes**: Intent is committed (J01 Commit Boundary Event) → ready for V1 Kernel
   - **No**: Memory is cleared, user starts over

## Switching Providers

Just change `LLM_PROVIDER` in your `.env` file:

```env
# Use Gemini:
LLM_PROVIDER=gemini

# Use OpenAI:
LLM_PROVIDER=openai
```

No code changes needed. The factory function in `llm_client.py` handles the rest.

## Project Structure

```
IntentReformulator/
├── prompts/
│   └── intent_extraction.yaml     # System prompt template
├── src/
│   ├── __init__.py
│   ├── memory.py                  # Sliding-window chat history manager
│   ├── llm_client.py             # LLM wrappers (Gemini + OpenAI) + factory
│   └── reformulator.py           # Intent proposal logic + validation
├── tests/
│   ├── __init__.py
│   └── test_multiturn.py         # Golden test cases (both providers)
├── docs/
│   └── DOCUMENTATION.md          # Full technical documentation
├── .env.example                  # Environment variable template
├── .gitignore
├── requirements.txt
├── README.md
└── main.py                       # Interactive CLI entry point
```
