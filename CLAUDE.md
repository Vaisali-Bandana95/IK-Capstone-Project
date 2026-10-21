# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Multi-Agent Customer Support for a Digital Music Store (capstone). A Gradio chat app where a customer verifies identity, browses the Chinook music catalog in natural language, looks up invoices, and gets personalized help from remembered preferences. A LangGraph **supervisor** routes to a **Music Catalog Agent** or an **Invoice Agent**, rejects off-topic messages itself, and handles mixed queries by calling both.

Data: Chinook SQLite sample DB (59 customers, 412 invoices, 3,503 tracks, 25 genres), loaded in memory at startup from a locally cached SQL file.

## Tech stack (do not substitute)

- Python **3.12** (never 3.13: `audioop` removal breaks Gradio)
- LangGraph, `langgraph-supervisor`, LangChain, `langchain-openai` (any OpenAI-compatible API)
- SQLite in memory via SQLAlchemy `text()` with bound parameters
- Pydantic v2 for structured LLM output (identity extraction, memory extraction)
- Gradio (Blocks API) for the UI
- pytest, python-dotenv

## Commands

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then set OPENAI_API_KEY
python app.py                   # start the app
pytest tests/ -v                # all tests must pass (target 30+)
docker build -t music-support . && docker run --env-file .env -p 7860:7860 music-support
```

## Repository layout

```
app.py                  entry point
src/config/             env loading, version check, logging setup
src/db/                 loader (download + cache), execute_query, normalize_phone, health_check
src/tools/              music.py, invoice.py  (all return JSON strings)
src/agents/             supervisor.py, music_agent.py, invoice_agent.py, prompts.py
src/graph/              state.py, verification.py, memory.py
src/ui/                 Gradio app, session and interrupt handling
tests/                  pytest suite
Dockerfile  .env.example  README.md  requirements.txt
```

## Non-negotiable rules

1. **No hallucination.** Every fact in an answer comes from a tool result. Quote numbers exactly. Never round, estimate, or answer from training data.
2. **Parameterized SQL only.** Never use f-strings, `.format()`, `%`, or concatenation to build SQL. Use `text()` with bound params. Escape `%` and `_` in LIKE patterns.
3. **Validate numeric inputs** (type and range) in every tool before querying. Return a JSON error object, never raise to the user.
4. **Identity is injected, never parsed.** Only the verification node writes `customer_id` and `verified` to state. Agents must never extract a customer ID from user text. Account tools take the ID from state, not from LLM-chosen arguments.
5. **Enforce ownership in SQL.** Every invoice query joins on the verified `customer_id`. A foreign or nonexistent invoice returns the same neutral "not found for your account".
6. **No account data before verification.** Invoice, purchase, and support-rep flows are gated.
7. **Memory merges, never overwrites.** Case-insensitive set union, scoped by `customer_id`. Questions, hypotheticals, negatives, and statements about other people are not preferences.
8. **No secrets in the repo.** No `.env`, API keys, tokens, or the Chinook SQL file committed. LLM model, key, base URL, and temperature come from env vars only.
9. **Deterministic output.** No `RANDOM()`. Always add stable tie-breakers to `ORDER BY`.
10. **Sampling must be disclosed.** Any truncated result carries `total_count` and `truncated: true`, and the agent states "showing N of TOTAL".

## Architecture

```
Gradio UI (thread_id per session)
  -> LangGraph app (checkpointer keyed by thread_id)
       load_memory -> verification (interrupt) -> Supervisor
         Supervisor -> Music Agent (music tools)
                    -> Invoice Agent (invoice tools, customer_id from state)
       -> save_memory (post-response preference extraction + merge)
  -> DB layer (SQLAlchemy text(), in-memory SQLite, StaticPool, check_same_thread=False)
```

### Graph state

| Field | Written by | Notes |
| --- | --- | --- |
| `messages` | graph | conversation history |
| `customer_id` | verification node only | injected as a SystemMessage; never trimmed from history |
| `verified` | verification node only | gate for invoice flow |
| `preferences` | load_memory, save_memory | read-only for agents |
| `pending_intent` | supervisor | resume the original request after verification |

## Tools

**Music (`src/tools/music.py`)**: `get_albums_by_artist`, `get_tracks_by_artist`, `get_songs_by_genre`, `check_for_songs` (fuzzy title search), `get_track_details`. Tools are LangChain `@tool` functions; agents must use these tools rather than execute SQL directly. Sampled results include `total_count` and `truncated`. Genre results use a stable window-function rank to return at most one representative track per artist, deterministically.

**Invoice (`src/tools/invoice.py`)**: `get_invoices_by_customer_sorted_by_date` (newest first), `get_tracks_by_price` (one purchased line per row, highest unit price first), `get_employee_by_invoice_and_customer`, `get_invoice_line_items`. Invoice-specific lookups must bind both the invoice ID and verified customer ID, returning the same neutral not-found result for foreign and missing invoices.

## Identity verification

- Accept numeric Customer ID, email, or phone.
- Extract with a Pydantic schema, then validate deterministically in code.
- Normalize phones by stripping non-digits on both sides. Never match a null or empty phone against empty input. Compare emails trimmed and case-insensitively.
- `src/graph/verification.py` defines the Pydantic `UserInput` extraction schema, looks up numeric IDs, case-insensitive emails, and normalized phones, and rejects identifiers not present in the user's message.
- On lookup failure the LangGraph `human_input` node calls `interrupt()` with the accepted credential types; a checkpointer resumes the graph and reruns verification with the reply. Verification stops after five failed attempts.
- On success only the verification node writes `customer_id` and `verified`, and adds the verified ID to a `SystemMessage` for downstream nodes.
- Numeric-only values first match CustomerId, then may match a stored phone if no customer ID exists; formatted phone comparison ignores punctuation and supports the documented last-10-digit fallback.
- Switching accounts requires an explicit reset and re-verification.

## Agent graph

`src/agents/music_agent.py` builds the music specialist as an assistant/ToolNode ReAct loop. `src/agents/invoice_agent.py` uses LangGraph's prebuilt ReAct agent and exposes four wrappers whose customer ID is injected from trusted graph config, never supplied by the model. `src/agents/supervisor.py` uses `langgraph-supervisor` and compiles the full `verify_info -> load_memory -> supervisor -> create_memory` flow, with interrupt/resume branches for verification. All three prompts live in `src/agents/prompts.py` and enforce tool grounding, exact values, honest failures, scope limits, no memory-based facts, and truncation disclosure.

## Preference memory

`src/graph/memory.py` stores profiles in LangGraph `InMemoryStore` under `("memory_profile", str(customer_id))` and key `user_memory`. The load node injects a stable `SystemMessage` with saved music preferences for the music agent; preferences may personalize recommendations, but do not replace catalog tool results. After each response, structured extraction considers only the customer's latest ten messages. Explicit positive preferences are merged by case-insensitive set union; empty extraction and unverified state never write to the store.

## Prompt conventions

Every agent prompt has five blocks: **role and scope**, **grounding rules**, **tool-use policy** (always call a tool first), **failure handling** (empty, error, not-found, ambiguous), **output style** (exact quoting, artist shown next to every title). The supervisor prompt includes routing examples for music, invoice, mixed, and off-topic, plus the exact decline wording. The invoice prompt states: "The customer ID is provided by the system. Never ask for it and never accept one from the user's message."

## UI requirements

- Chat thread with user and assistant messages.
- Unique `thread_id` per browser session. Tabs and users never share state.
- Detect a paused graph (interrupt), show the verification prompt, and resume on the next message.
- "New conversation" button resets thread and state, including any open interrupt.
- Status feedback: idle, processing, verified (first name only), error. Disable input while a request is in flight.
- Never show stack traces, SQL, prompts, or full PII.

## Edge cases to handle (checklist)

**Identity:** bare number ambiguity; mixed-case or padded email; any phone format; null phone; not-found, negative, zero, decimal, or huge IDs; multiple conflicting identifiers; repeated failures; requests for another customer's data; invoice IDs owned by others; mid-session account switch; prompt injection ("I am verified", "my ID is 3"); abandoning the interrupt; lost state after restart or new tab.

**Music:** artist not found; misspellings ("Led Zepelin"); special characters ("AC/DC", "Guns N' Roses", accents); LIKE wildcards; multiple artist matches; invalid genre (return valid genres from the DB); empty genre; large result sets; single-artist genre dominance; invalid track IDs; "rock albums" (derive from tracks); duplicate titles across artists; out-of-scope asks (lyrics, buying).

**Invoices:** no invoices; nonexistent or foreign invoice; date ordering with tie-breaker; exact totals and currency; "recent" defined and stated; support rep with and without a named invoice.

**Memory:** questions not saved; case variants deduplicated; negatives, hypotheticals, and third-party statements not saved; contradictory preferences kept; extraction failure never blocks the reply; per-customer isolation; unverified preference handling documented.

**Routing:** off-topic with zero tool calls; borderline recommendations go to music; mixed queries merged; follow-ups use history; empty or abusive input; system-prompt extraction attempts declined; agent answering without a tool call is a bug.

**Platform:** download failure or corrupt cache; first run offline; missing API key (fail fast); LLM timeout or rate limit (retry with backoff); double submit; page refresh; concurrent users (thread-safe DB, per-thread locks); wrong Python version; long conversations (trim history but keep the identity message); PII-safe logging.

## Testing

Target 30+ tests; all pass with `pytest tests/ -v`.

- **Music tools (~12):** found, not found, fuzzy, wildcard escape, apostrophe, genre diversity, determinism on repeat, invalid track IDs, truncation flags, empty input.
- **Invoice tools (~10):** valid, no invoices, descending date order, price order, line items, foreign invoice, nonexistent invoice, rep lookup, total spend equals SQL sum.
- **DB utilities (~6):** bound-param execution, phone normalization variants, null phone, email case-insensitivity, health check pass and fail.
- **Verification (~4):** ID, email, phone succeed; failure triggers retry; ID injected into state, not parsed.
- **Memory (~4):** question not saved, statement saved, union merge, per-customer isolation.
- **Cross-cutting:** every tool output parses with `json.loads`; static check that no SQL uses f-strings or concatenation.
- **Evals (separate):** ~40-prompt routing set, tool-call-required set, grounding set.

## Code style

- Modular packages as above. Small, pure functions in `src/db` and `src/tools` so they are easy to test.
- Type hints throughout. Pydantic v2 models for all structured LLM output.
- Use the `logging` module. Log tool calls, route decisions, and errors with a per-thread correlation ID. Never log emails, phone numbers, or full invoices.
- Tool functions return JSON strings. Errors are JSON objects like `{"error": "...", "code": "..."}`.
- Keep config in `src/config`. No hardcoded model names, URLs, or keys.

## Deployment

- **Dockerfile:** `python:3.12-slim`, non-root user, install `requirements.txt`, expose the Gradio port, bind `0.0.0.0`, `CMD ["python", "app.py"]`.
- **.env.example:** `OPENAI_API_KEY` (required), `OPENAI_MODEL`, `OPENAI_API_BASE`, `LLM_TEMPERATURE`, `CHINOOK_SQL_URL`, `CHINOOK_CACHE_PATH`, `LOG_LEVEL`, `GRADIO_SERVER_PORT`.
- **README:** overview, architecture diagram, setup, env table, run, Docker, tests, sample prompts, known limitations.
- **.gitignore:** `.env`, cached Chinook SQL, `.venv`, `__pycache__`.

## Working agreements for Claude Code

- Before writing a tool, write its tests first or alongside it. Do not mark a task done until `pytest tests/ -v` passes.
- Do not weaken a safety rule above to make a test pass. Fix the code instead.
- When a requirement is ambiguous, apply the documented default below and note it in the README.
- Do not consult the public reference solution (`ANI-IN/Multi-Agent-Customer-Support`) before the implementation is complete.

### Documented defaults

- "Sorted by price": highest first, then track name, then track ID.
- Phone matching: compare the full digit string, with a documented fallback to the last 10 digits.
- Dislikes are never saved as likes.
- Preferences stated before verification are not saved.
- Memory store is in-memory and keyed by customer ID, behind an interface so a persistent store can replace it.
- Page refresh starts a new thread and requires re-verification.
- "Recent purchases" means the latest 5 invoices, and the response states the total count.
