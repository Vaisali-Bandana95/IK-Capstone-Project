# Chinook Music Support

A Gradio support chat for the Chinook digital music store dataset. Customers
verify their identity before accessing invoice or purchase information. A
supervisor routes catalog requests to a music agent and account requests to an
invoice agent. Catalog and account facts are returned by parameterized database
tools; verified customer preferences can personalize suggestions.

## Requirements

- Python 3.12
- An OpenAI-compatible API key and model for live LLM conversations
- Docker is optional

## Setup

Create and activate a Python 3.12 virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` and `OPENAI_MODEL` in `.env`. `OPENAI_API_BASE` is optional
for OpenAI-compatible endpoints. The SQL source is downloaded from the URL in
`CHINOOK_SQL_URL` once and cached locally at `CHINOOK_CACHE_PATH`. Both settings
can be overridden in `.env`.

## Run

```powershell
python app.py
```

Open `http://localhost:7860` (or the configured `PORT`). Each browser session is
assigned a UUID thread ID. Use **New conversation** to clear the chat and start a
new verification flow; preference memory remains associated with the verified
customer in the current application process.

## Run with Docker

```powershell
docker build -t chinook-music-support .
docker run --env-file .env -p 7860:7860 chinook-music-support
```

The container runs on Python 3.12, listens on `0.0.0.0:7860`, and downloads the
Chinook SQL source into its local cache on first startup. The `.env` file is
excluded from the Docker build context.

## Deploy to Render

The repository includes a Render Blueprint in `render.yaml`. To deploy:

1. Open the [Render Blueprint creation page](https://dashboard.render.com/blueprint/new)
   and connect this GitHub repository.
2. Select the `main` branch and review the `chinook-music-support` web service.
3. Enter `OPENAI_API_KEY` and `OPENAI_MODEL` when Render prompts for them.
   Keep secrets in Render's environment settings; never commit them to the repo.
4. Create the Blueprint and wait for the Docker build and health check to finish.
5. Open the `onrender.com` URL shown on the service page.

The service binds to Render's `PORT` value and uses `/` for its health check.
The Free plan can spin down while idle, so the first request after inactivity
may take longer. Preference memory and the downloaded SQL cache are in the
container's ephemeral storage; memory is per process and does not survive a
restart or redeploy.

## Test

```powershell
python -m pytest tests/ -v
```

Most tests use deterministic local fake models and do not call the configured
LLM API. They cover SQL tools, identity verification and interrupt/resume,
supervisor routing, session-scoped graph state, and preference memory.

## Sample usage

1. Start with a Customer ID, email, or phone number when asked to verify.
2. Ask: “What rock albums do you have?”
3. Ask: “Show me my invoices.”
4. State a preference, for example: “I love jazz.”
5. Ask for music suggestions in a later turn; the saved preference can inform
   recommendations, while catalog facts still come from music tools.
6. Select **New conversation** to reset the chat. A new thread requires identity
   verification again.

## Architecture

```text
Gradio Blocks UI
  └─ per-session UUID thread_id
      └─ LangGraph support graph (checkpointer)
          ├─ verify_info
          │   └─ human_input interrupt / resume
          ├─ load_memory (InMemoryStore per verified customer)
          ├─ supervisor (langgraph-supervisor)
          │   ├─ music agent (assistant ↔ catalog ToolNode)
          │   └─ invoice agent (prebuilt ReAct agent)
          └─ create_memory (structured preference extraction and merge)
```

Invoice tools receive the verified customer ID from trusted graph runtime
configuration, not an LLM-selected argument. Preference profiles are isolated
in memory by customer ID, merge by case-insensitive set union, and are lost when
the process restarts.
