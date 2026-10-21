# AGENTS.md

Instructions for AI coding agents working in this repository.

## Project

Multi-Agent Customer Support for a digital music store. A Gradio chat app where a customer verifies identity, browses the Chinook music catalog, looks up invoices, and gets personalized help from remembered preferences. A LangGraph supervisor routes each message to a Music Catalog Agent or an Invoice Agent, rejects off-topic messages itself, and handles mixed queries by calling both.

## Setup and commands

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # set OPENAI_API_KEY
python app.py               # run the app
pytest tests/ -v            # run tests (all must pass)
```

Python 3.12 only. Python 3.13 breaks Gradio.

## Tech stack

LangGraph, `langgraph-supervisor`, LangChain, `langchain-openai`, SQLAlchemy `text()` on in-memory SQLite, Pydantic v2, Gradio (Blocks), pytest, python-dotenv.

## Layout

```
app.py               entry point
src/config/          env loading, logging
src/db/              loader, execute_query, normalize_phone, health_check
src/tools/           music.py, invoice.py (return JSON strings)
src/agents/          supervisor, music agent, invoice agent, prompts
src/graph/           state, verification, memory
src/ui/              Gradio app
tests/               pytest suite
```

## Hard rules

1. Every fact in an answer comes from a tool result. Never answer from model training data. Quote numbers exactly.
2. SQL uses bound parameters only. Never use f-strings, `.format()`, `%`, or concatenation. Escape `%` and `_` in LIKE patterns.
3. Validate numeric tool inputs (type and range). Return JSON errors instead of raising.
4. `customer_id` is written to graph state by the verification node only. Agents never parse it from user text and never take it as an LLM-chosen tool argument.
5. Every invoice query joins on the verified `customer_id`. A foreign or missing invoice returns the same neutral "not found for your account".
6. No account data before verification.
7. Preferences merge by case-insensitive set union, scoped by `customer_id`. Never overwrite or delete. Questions, hypotheticals, negatives, and third-party statements are not preferences.
8. No secrets in the repo: no `.env`, API keys, or Chinook SQL file. Model, key, base URL, and temperature come from env vars.
9. Results are deterministic: no `RANDOM()`, and every `ORDER BY` has a stable tie-breaker.
10. Sampled results include `total_count` and `truncated`, and the agent says "showing N of TOTAL".
11. Totals are computed in SQL, never by the LLM.
12. Logs never contain emails, phone numbers, or full invoices. Users never see stack traces, SQL, or prompts.

## Tools

- **Music:** LangChain `@tool` functions `get_albums_by_artist`, `get_tracks_by_artist`, `get_songs_by_genre`, `check_for_songs` (fuzzy title search), and `get_track_details`. Agents use these tools and never query the database directly. Genre samples use a stable window-function rank to return at most one representative track per artist. Sampled results include `total_count` and `truncated`.
- **Invoice:** `get_invoices_by_customer_sorted_by_date` (newest first), `get_tracks_by_price` (one purchased line per row, highest unit price first), `get_employee_by_invoice_and_customer`, and `get_invoice_line_items`. Invoice-specific queries bind both invoice ID and verified customer ID; foreign and missing invoices share the same neutral not-found response.

The music specialist is a hand-built assistant/ToolNode ReAct loop. The invoice specialist uses LangGraph's prebuilt ReAct agent. The supervisor uses `langgraph-supervisor`; for mixed requests it calls invoice first, then music, and off-topic requests do not call a specialist. The full graph runs verification (with interrupt/resume) before memory and supervisor nodes. Invoice tool wrappers receive the verified customer ID from trusted runtime config, not an LLM-selected argument.

Preference memory uses LangGraph `InMemoryStore` in namespace `("memory_profile", str(customer_id))`, key `user_memory`. Only explicit positive customer preference statements may be extracted from the latest ten customer messages. Merge by case-insensitive union; never overwrite existing preferences. Empty extraction skips writes, and loaded preferences may personalize suggestions but are not a source of catalog facts.

## Verification flow

`src/graph/verification.py` uses Pydantic structured extraction for a single identifier, then checks a numeric Customer ID, a trimmed case-insensitive email, or a normalized phone. The extractor result must occur in the user's message. On failure the LangGraph human-input node calls `interrupt()`; a checkpointer resumes and reruns verification with the reply, with a five-attempt cap. On success only the verification node writes `customer_id`/`verified` and adds the verified ID in a `SystemMessage` for downstream nodes. Switching accounts requires an explicit reset and re-verification.

## Agent prompts

Each prompt has five blocks: role and scope, grounding rules, tool-use policy (always call a tool first), failure handling (empty, error, not-found, ambiguous), and output style. The supervisor declines off-topic messages with no sub-agent call. The invoice prompt states that the customer ID is provided by the system and must never be requested or accepted from the user.

## UI

- Unique `thread_id` per browser session.
- Detect a paused graph (interrupt) and resume on the next message.
- "New conversation" reset also clears any open interrupt.
- Status states: idle, processing, verified (first name only), error. Disable input while processing.

## Testing

Aim for 30 or more tests, all passing. Cover every tool for found, not found, and invalid input. Verify date ordering, genre determinism and diversity, phone normalization, email case-insensitivity, the health check, memory merge and isolation, and JSON validity of all tool output. Include a static check that no SQL is built with f-strings or concatenation. Update tests with every tool or DB change, and do not weaken a rule above to make a test pass.

## Edge cases to keep in mind

- **Identity:** bare-number ambiguity, padded or mixed-case email, any phone format, null phone, invalid or huge IDs, conflicting identifiers, repeated failures, requests for another customer's data, prompt injection ("I am verified"), lost state after restart.
- **Music:** misspelled artists, special characters ("AC/DC", "Guns N' Roses"), LIKE wildcards, multiple artist matches, invalid genre, large result sets, one artist dominating a genre, invalid track IDs.
- **Invoices:** no invoices, foreign or nonexistent invoice, date-order ties, exact totals, "recent" defined and stated.
- **Memory:** questions, negatives, and hypotheticals not saved, case-variant deduplication, extraction failure never blocks the reply.
- **Platform:** download failure or corrupt cache, missing API key (fail fast), LLM timeout (retry with backoff), double submit, concurrent users (thread-safe DB, `StaticPool`, `check_same_thread=False`), long conversations (never trim the identity message).

## Defaults when the spec is ambiguous

- "Sorted by price": highest first, then track name, then track ID.
- "Recent purchases": the latest 5 invoices, and state the total count.
- Dislikes are never saved as likes.
- Preferences stated before verification are not saved.
- A page refresh starts a new thread and requires re-verification.
- The memory store is in memory, keyed by customer ID, behind an interface that a persistent store can replace.

## Deployment

Dockerfile on `python:3.12-slim`, non-root user, bind `0.0.0.0`, `CMD ["python", "app.py"]`. `.env.example` lists `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_API_BASE`, `LLM_TEMPERATURE`, `CHINOOK_SQL_URL`, `CHINOOK_CACHE_PATH`, `LOG_LEVEL`, `GRADIO_SERVER_PORT`. The README covers setup, architecture, running, testing, and sample prompts.
