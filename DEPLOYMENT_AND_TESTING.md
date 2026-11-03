# Capstone: deployment and acceptance test instructions

## What changed

- Explicit supervisor conditional edges for catalog, invoice, mixed (invoice first), identity lookup and off-topic rejection.
- Verified customer ID comes from graph state and is passed through trusted RunnableConfig, never from the agent's tool arguments.
- Ten tools cover catalog and invoice lookups. Album track-list requests return every matching track with its artist. Genre totals mean total **tracks**, with separate album and artist counts; representative samples remain diverse.
- Added typo-tolerant artist/song/genre search using Python's standard library and parameterized SQL.
- Invoice list contains an SQL-calculated `total_spent` field suitable for exact spending responses; no LLM addition of floats.
- Per-customer preferences now persist in SQLite at MEMORY_DB_PATH and are merged without overwriting older preferences.
- LangGraph conversation and verification checkpoints persist in SQLite at CHECKPOINT_DB_PATH.
- Added tests for off-topic rejection, anonymous catalog browsing, verification gating and resume, mixed routing, incorrect IDs, catalog counts, spending totals, fuzzy matching and preference persistence.
- Docker Compose stores preferences, checkpoints and the cached SQL in the
  persistent `chinook-data` named volume mounted at `/data`.

## Local Docker Desktop deployment

1. Open a PowerShell terminal in the project folder.
2. Run `Copy-Item .env.example .env` if `.env` is missing.
3. Set `OPENAI_API_KEY` and `OPENAI_MODEL` in `.env` (a model that supports tool calling and structured output).
4. Run `docker compose up --build -d`.
5. Check `docker compose logs --tail 100 -f` and open http://127.0.0.1:7861.
6. To stop without losing data, run `docker compose down`. The named `chinook-data` volume is retained. Do **not** run `docker compose down -v` unless you intend to delete its stored data.
7. Check persistence: log in as customer 5, say `I love jazz and AC/DC`, start a new conversation, enter customer ID 5 and ask for recommendations. Restart the container, repeat and confirm preferences remain. An interrupted conversation with the same thread ID is also retained in the checkpoint database.

If data bootstrap fails, verify outbound HTTPS access to GitHub raw content (`CHINOOK_SQL_URL`). The sample catalog is loaded into in-memory SQLite on each process start. The source SQL is cached in `app/cache` locally or under `/data` when that directory is backed by persistent storage, such as the Compose named volume. The catalog data is read-only sample data.

## Python tests (same version as Docker: 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

`tests/test_capstone_acceptance.py` includes 9 new checks. The full suite requires one-time access to the Chinook SQL source unless `CHINOOK_CACHE_PATH` already points to its cached file. There is no automated live LLM evaluation; all six scenarios still need manual chat testing with a valid model key.

## Render hosting

`render.yaml` uses **Free** and has no persistent disk. Render Free can remove
local SQLite files on spin-down or redeploy, so it cannot guarantee preference
or checkpoint persistence. Set `OPENAI_API_KEY` and `OPENAI_MODEL` in the
dashboard. For durable local/Docker testing, use the Compose named volume. To
make Render persistence durable, select a paid plan that supports a disk and
configure the disk in the dashboard; confirm pricing before enabling it.

## Demo-only authentication caveat

The capstone rubric calls matching a customer ID/email/phone in the supplied sample database "verification." This project implements **that lookup gate**, but these publicly knowable identifiers do not prove real identity. **Do not deploy with real customer records or real invoices without proper authentication (e.g. an authenticated session or one-time email/phone code).**

## Limits of the provided patch

The dependency versions are pinned in `requirements.txt`. Run `pip check`, the
full test suite, and a Docker Compose build on a networked machine. The source
code alone cannot guarantee an external LLM always classifies or formats an
answer correctly; do the full manual acceptance demo before submission.
