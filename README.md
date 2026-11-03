# Chinook Music Support — Capstone patch

A Gradio chat for the Chinook digital music database. A LangGraph supervisor classifies incoming requests and routes them to scoped catalog and invoice agents using explicit conditional graph edges. Ten SQLAlchemy tools cover catalog searches (including complete track lists by album) and invoice lookups.

> **Important:** Matching a customer ID, email or phone to the demo Chinook database is the capstone's *lookup-based identity gate*, **not** secure authentication. Do not upload real customer records or expose private invoices on the public internet without adding a proper authentication mechanism.

## Requirements

- Python 3.12, or Docker Desktop with Docker Compose
- An OpenAI-compatible API key and tool-calling + structured-output-capable model
- Internet access for first-run download of the public Chinook SQL dataset (or pre-populate the cache)

## Quick start: Docker Desktop (preferred)

```powershell
Copy-Item .env.example .env
# Edit .env and set OPENAI_API_KEY and OPENAI_MODEL.
docker compose up --build -d
docker compose logs --tail 100 -f
```

Then visit <http://127.0.0.1:7861>. Docker Compose maps host port 7861 to the
container's Gradio port 7860 and stores customer preferences,
conversation checkpoints, and the downloaded Chinook SQL in a named volume so
they survive container restarts and recreation. Stop with `docker compose down`;
the named volume is retained. To remove it and its data intentionally, run
`docker compose down -v`.

## Python local development

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
# Set the API key and model in .env.
.\.venv\Scripts\python.exe app.py
```

## Render deployment

`render.yaml` targets Render's **Free** plan. Set `OPENAI_API_KEY` and
`OPENAI_MODEL` using Render's Blueprint setup form or environment variables.
Render Free does not provide a persistent disk: local SQLite preferences,
conversation checkpoints, and the SQL cache can be lost when the service spins
down or redeploys. Use Docker Compose with its named volume for durable local
storage. To retain data on Render, choose a paid plan that supports a persistent
disk and configure that disk in the Render dashboard.

See [DEPLOYMENT_AND_TESTING.md](DEPLOYMENT_AND_TESTING.md) for precise commands, manual demo acceptance checks, failure diagnosis and deployment limitations.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

Most test cases use deterministic fake models, so no real LLM usage is required. Integration tests need the public Chinook SQL dataset on first execution. GitHub Actions CI (`.github/workflows/ci.yaml`) installs the pinned dependencies, runs tests, and builds the image.

## Expected scenarios

1. Ask to see invoices -> app prompts for sample customer ID/email/phone -> a correct match permits the account-scoped invoice specialist.
2. Ask for rock albums -> supervisor goes directly to the catalog specialist -> show an artist-diverse sample and distinct album/track/artist counts.
3. After lookup, ask about purchases -> the invoice specialist uses the customer ID from graph state, not from user text.
4. State `I really love jazz and AC/DC` while verified -> preferences are merged per customer; they are reloaded in later conversations, including after a Docker restart with the persistent volume.
5. Ask `What's the weather today?` -> supervisor directly rejects the request, with no specialist called.
6. Enter customer ID `99999` -> lookup fails -> graph pauses and asks for another ID/email/phone.
7. Ask both invoice and catalog questions in one message -> explicit sequential invoice then music specialist execution.
8. Ask how much you spent -> the invoice list tool returns a SQL-calculated `total_spent` field; the LLM is told not to calculate totals itself.

## Architecture

```text
Gradio Blocks UI (session UUID, interrupt/resume handling)
  └─ LangGraph supervised support graph
       ├─ supervisor route/classify
       │   ├─ off_topic -> direct refusal (END)
       │   ├─ verify -> customer lookup -> customer-specific memory
       │   ├─ music -> catalog specialist -> reply
       │   ├─ invoice -> verified ID check -> invoice specialist -> reply
       │   └─ mixed -> verified ID check -> invoice specialist -> music specialist -> reply
       ├─ account identification by ID/email/phone with retry interrupts
       ├─ preferences in SQLite
       └─ conversation checkpoints in SQLite

Music specialist: six LangChain tools -> parameterized Chinook SQL
Invoice specialist: four LangChain tools -> verified customer ID injected by RunnableConfig
```

The Chinook music catalog is read-only and loaded into an in-memory SQLite
database at startup. The source SQL is cached separately. Customer preferences
and LangGraph conversation checkpoints use separate persistent SQLite
databases. With Docker Compose, all three files are stored in the `chinook-data`
named volume. Render Free's filesystem is ephemeral, so it does not guarantee
persistence.
