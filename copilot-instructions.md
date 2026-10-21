# Copilot Instructions

Project: Multi-Agent Customer Support for a digital music store. A Gradio chat app. A LangGraph supervisor routes to a Music Catalog Agent or an Invoice Agent. Data comes from the Chinook SQLite database, loaded in memory.

## Stack

- Python 3.12 only. Never suggest 3.13 (Gradio breaks without `audioop`).
- LangGraph, `langgraph-supervisor`, LangChain, `langchain-openai`
- SQLAlchemy `text()` with SQLite in memory
- Pydantic v2, Gradio (Blocks), pytest, python-dotenv

## Commands

- Run: `python app.py`
- Test: `pytest tests/ -v`
- Install: `pip install -r requirements.txt`

## Layout

- `src/config/`: env loading, logging
- `src/db/`: loader, `execute_query`, `normalize_phone`, `health_check`
- `src/tools/`: `music.py`, `invoice.py` (return JSON strings)
- `src/agents/`: supervisor, music agent, invoice agent, prompts
- `src/graph/`: state, verification, memory
- `src/ui/`: Gradio app
- `tests/`: pytest suite

## Always

- Use parameterized queries: `text("... WHERE id = :id")` with bound params.
- Escape `%` and `_` in LIKE patterns.
- Validate every numeric tool input (type and range) before querying.
- Return tool results as JSON strings. Return errors as `{"error": "...", "code": "..."}`.
- Add stable tie-breakers to every `ORDER BY`.
- Include `total_count` and `truncated` in any sampled result.
- Take `customer_id` from graph state, set only by the verification node.
- Join on the verified `customer_id` in every invoice query (ownership check in SQL).
- Merge preferences with a case-insensitive set union, keyed by `customer_id`.
- Read the model name, API key, base URL, and temperature from environment variables.
- Use the `logging` module, with type hints and Pydantic v2 models for structured LLM output.
- Write or update tests with every tool or DB change.
- Agents access the database only through the nine LangChain `@tool` functions in `src/tools/`; never add raw SQL execution to an agent.
- The music specialist uses a hand-built assistant/ToolNode ReAct loop; the invoice specialist uses the prebuilt ReAct agent; the supervisor uses `langgraph-supervisor`.
- Preserve the `verify_info -> load_memory -> supervisor -> create_memory` graph flow. Invoice customer identity must be injected through trusted runtime config from verified state, never exposed as a model-selected tool argument.
- Store preferences in LangGraph `InMemoryStore` per verified customer; extract only explicit positive statements from customer messages and merge via case-insensitive set union. Empty extraction must not write or erase existing preferences.

## Never

- Build SQL with f-strings, `.format()`, `%`, or concatenation.
- Let an agent read a customer ID from user text, or accept one as an LLM-chosen tool argument.
- Return account data before verification.
- Reveal whether an invoice that belongs to another customer exists.
- Use `RANDOM()` in queries (results must be deterministic).
- Save questions, hypotheticals, negatives, or third-party statements as preferences.
- Overwrite or remove saved preferences.
- Do arithmetic in the LLM. Compute totals in SQL.
- Hardcode API keys, model names, or URLs. Never commit `.env` or the Chinook SQL file.
- Log emails, phone numbers, or full invoices.
- Show stack traces, SQL, or prompts to the user.

## Agent prompts

Every agent prompt must contain these five blocks:

1. Role and scope (what it covers and what it defers)
2. Grounding rules (tool-only facts, exact numbers, honest empty results, never use training data)
3. Tool-use policy (always call a tool before answering a factual question)
4. Failure handling (empty, error, not-found, ambiguous)
5. Output style (quote values exactly, show the artist next to every title, state "showing N of TOTAL" when truncated)

The supervisor declines off-topic messages itself and calls no sub-agent. The invoice prompt must say the customer ID is provided by the system and must never be requested or accepted from the user.

## Verification

- Accept a numeric Customer ID, an email, or a phone number.
- Strip non-digits from both sides before comparing phones. Never match a null or empty phone.
- Compare emails trimmed and case-insensitively.
- Extract one identifier with the Pydantic `UserInput` schema and reject any extracted value not present in the user's message.
- On failure, call LangGraph `interrupt()` with a retry prompt; use a checkpointer to resume and rerun verification, capped at five failed attempts.
- On success, only the verification node writes `customer_id` and `verified`, and injects the verified ID through a `SystemMessage`.

## UI

- One unique `thread_id` per browser session.
- Detect a paused graph and resume on the next message.
- Provide a "New conversation" reset that also clears any open interrupt.
- Show idle, processing, verified (first name only), and error states. Disable input while processing.

## Tests

Target 30 or more passing tests:

- Every tool: found, not found, invalid input.
- Date ordering (newest first) and determinism of genre queries.
- Genre samples span multiple artists.
- Phone normalization variants, email case-insensitivity, health check.
- Memory: merge, question not saved, per-customer isolation.
- Every tool output parses with `json.loads`.
- Static check: no SQL built with f-strings or concatenation.

## Defaults when the spec is ambiguous

- "Sorted by price": highest first, then track name, then track ID.
- "Recent purchases": the latest 5 invoices, and state the total count.
- Dislikes are never saved as likes.
- Preferences stated before verification are not saved.
- A page refresh starts a new thread and requires re-verification.
