# AI Assistant Platform

A full-stack, multi-model AI assistant platform - chat across OpenAI, Anthropic Claude,
and Google Gemini; retrieval-augmented generation over your own documents; long/short-term
memory; tool & function calling; a real Model Context Protocol (MCP) server and client;
multi-step agents; and integrations with Google Calendar, Gmail, Telegram, voice, and vision.

Built with **FastAPI**, **SQLite**, **Pydantic v2**, and vanilla **HTML/CSS/JS** - no
frontend build step required.

---

## 1. Project overview

This project demonstrates an end-to-end AI engineering stack in a single, readable
repository:

| Layer | What it shows |
|---|---|
| Multi-model orchestration | `llm_providers.py` - one interface over 3 vendor SDKs |
| RAG | `rag_service.py` - chunk -> embed -> cosine search -> cited context |
| Memory | `memory_service.py` - windowed short-term + persisted long-term facts |
| Tool/function calling | `tools.py` + provider-specific tool-calling loops |
| Agents | `agents.py` - plan -> act -> observe loop with 3 example personas |
| MCP | `mcp_server.py` / `mcp_client.py` - real stdio MCP server + client |
| Integrations | Gmail, Calendar (Google OAuth2), Telegram bot, voice (Whisper/TTS), vision (GPT-4o/Claude vision) |
| Automation | Generic outbound webhook dispatcher for n8n / Zapier / Make |

## 2. Architecture

Clean, mostly-flat architecture - see [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full
breakdown. In short:

```
Browser (Jinja2 templates + vanilla JS)
        │  fetch() with JWT bearer token
        ▼
FastAPI routers (routes_*.py)  ──►  Service layer (chat_service, rag_service, ...)
        │                                   │
        ▼                                   ▼
   SQLAlchemy async ORM (models.py)   llm_providers.py (OpenAI / Anthropic / Gemini)
        │
        ▼
     SQLite (data/app.db)
```

## 3. Folder structure

```
ai-assistant-platform/
├── main.py # FastAPI app entrypoint
├── config.py # Settings (env vars)
├── database.py # Async SQLAlchemy engine/session
├── models.py # ORM models
├── schemas.py # Pydantic request/response models
├── auth.py # JWT + password hashing
├── llm_providers.py # Multi-model abstraction (OpenAI/Anthropic/Gemini)
├── chat_service.py # Chat orchestration (memory + RAG + tools)
├── memory_service.py # Short/long-term memory
├── rag_service.py # Document chunking/embedding/search
├── tools.py # Tool/function-calling implementations
├── code_assistant.py # Code generate/explain/debug/refactor/review/test
├── agents.py # Multi-step agent loop
├── mcp_server.py # MCP server (tools, resources, prompts)
├── mcp_client.py # MCP client demo
├── google_auth_service.py # Shared Google OAuth2 flow
├── gmail_service.py # Gmail integration
├── calendar_service.py # Google Calendar integration
├── telegram_service.py # Telegram bot
├── voice_service.py # Speech-to-text / text-to-speech
├── vision_service.py # OCR / captioning / receipt analysis
├── file_search_service.py # Metadata + semantic file search
├── automation_service.py # n8n / Zapier / Make webhooks
├── rate_limiter.py # In-memory rate limit middleware
├── logging_config.py # App-wide logging
├── routes_*.py # FastAPI routers, one per feature area
├── static/ # CSS + JS (no build step)
├── templates/ # Jinja2 pages
├── tests/ # pytest suite
├── docs/ # Additional docs
├── requirements.txt
├── .env.example
├── Start App.bat # Windows one-click launcher
└── Start App (Mac).command # macOS one-click launcher
```

## 4. Technology stack

- **Backend:** Python 3.12+, FastAPI, Pydantic v2, SQLAlchemy 2.0 (async), SQLite
- **AI SDKs:** `openai`, `anthropic`, `google-generativeai`, `langchain`
- **Auth:** JWT (PyJWT) + bcrypt password hashing
- **Frontend:** HTML5, CSS3 (custom design system, no framework), vanilla JS
- **Integrations:** Google API Python Client (Gmail/Calendar), `python-telegram-bot`, MCP SDK
- **Testing:** pytest, pytest-asyncio, httpx

## 5. Installation

### 5.1 Prerequisites
- Python 3.12 or newer
- Git
- (Optional) Visual Studio Code

### 5.2 Visual Studio Code setup
1. Install the **Python** extension (ms-python.python).
2. Open the repository folder: `File -> Open Folder...`
3. Select the interpreter: `Ctrl/Cmd+Shift+P` -> `Python: Select Interpreter` -> choose the
   `.venv` created below.
4. Recommended: install the **Even Better TOML** and **SQLite Viewer** extensions.

### 5.3 Python installation
Download Python 3.12+ from [python.org](https://www.python.org/downloads/) and verify:
```bash
python3 --version
```

### 5.4 Virtual environment
```bash
python3 -m venv .venv

# Activate:
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 5.5 Install dependencies
```bash
pip install -r requirements.txt
```

### 5.6 Configure environment variables
```bash
cp .env.example .env
```
Edit `.env` and fill in at least one AI provider key (see below). `APP_SECRET_KEY` and
`JWT_SECRET_KEY` should be replaced with long random strings for anything beyond local
testing.

### 5.7 Running FastAPI
```bash
uvicorn main:app --reload
```
Or use the one-click scripts:
- **Windows:** double-click `Start App.bat`
- **macOS:** double-click `Start App (Mac).command` (first run: `chmod +x` it, see script header)

Then open **http://localhost:8000**.

## 6. Configuring AI providers

### OpenAI
1. Create a key at https://platform.openai.com/api-keys
2. Set `OPENAI_API_KEY` in `.env`. Adjust `OPENAI_MODEL` if desired (default `gpt-4o-mini`).

### Claude (Anthropic)
1. Create a key at https://console.anthropic.com/
2. Set `ANTHROPIC_API_KEY` in `.env`. Adjust `ANTHROPIC_MODEL` if desired.

### Gemini (Google)
1. Create a key at https://aistudio.google.com/app/apikey
2. Set `GOOGLE_API_KEY` in `.env`. Adjust `GEMINI_MODEL` if desired.

You can configure one, two, or all three - the model switcher in the chat UI only shows
providers as "configured" once a key is present, and the app runs fine with any subset.

## 7. RAG (retrieval-augmented generation)

Upload a PDF, TXT, or Markdown file from the **Documents** page. The pipeline:
1. Extracts text (`pypdf` for PDFs)
2. Chunks it (~800 chars, 120 overlap)
3. Embeds each chunk (OpenAI `text-embedding-3-small` if `OPENAI_API_KEY` is set, otherwise
   a deterministic local hashing embedding so the feature still works with zero external
   dependencies - useful for demos without API keys)
4. Stores chunks + embeddings in SQLite
5. At query time, retrieves the top-K most similar chunks by cosine similarity and injects
   them into the system prompt with `[Source N]` citations

## 8. Memory

- **Short-term:** the last 20 messages of a conversation, sent verbatim as context.
- **Long-term:** durable facts (name, preferences, recurring constraints) that the app
  asks the LLM to identify after each user message; stored in `memory_items` and injected
  into every future conversation's system prompt. You can also add/edit/delete memories
  manually from the **Memory** page.

## 9. MCP (Model Context Protocol)

`mcp_server.py` is a real, standalone MCP server exposing:
- **Tools:** `web_search`, `calculator` (same implementations used by in-app chat)
- **Resources:** `platform://about`, `platform://changelog`
- **Prompts:** `summarize_document`, `code_review`

Run the server directly, or point any MCP-compatible client (Claude Desktop, another
agent, etc.) at it:
```bash
python mcp_server.py
```

`mcp_client.py` is a working demo client that launches the server as a subprocess over
stdio and exercises all three primitives:
```bash
python mcp_client.py
```

> **Note:** stdio-based MCP servers reserve **stdout** exclusively for JSON-RPC protocol
> messages. `mcp_server.py` therefore logs to **stderr** only - don't route its logging
> through the shared stdout logger used by the web app.

## 10. Automation (n8n / Zapier / Make)

Set `N8N_WEBHOOK_URL`, `ZAPIER_WEBHOOK_URL`, and/or `MAKE_WEBHOOK_URL` in `.env`, then call:
```
POST /api/automation/{platform}/trigger?event=my_event
Body: { "any": "json payload" }
```
This POSTs `{"event": ..., "payload": ...}` to your configured webhook, letting you trigger
external workflows from anywhere in the app.

## 11. Voice

- **Speech-to-text:** Whisper via the OpenAI API (`/api/voice/transcribe`)
- **Text-to-speech:** OpenAI TTS (`/api/voice/speak`)

Both require `OPENAI_API_KEY`. Try them on the **Voice** page (uses your browser's
microphone via `MediaRecorder`).

## 12. Vision

OCR, image captioning, and receipt analysis use GPT-4o (default) or Claude's vision
capabilities. Try them on the **Vision** page.

## 13. Testing

```bash
pytest
```
The suite uses an in-memory SQLite database and httpx's ASGI transport, so it needs no
running server and no API keys. It covers auth, conversation/memory/prompt CRUD, RAG
chunking/embedding math, and the tool-calling safety guardrails.

## 14. Deployment

For a simple single-process deployment:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```
Put this behind a reverse proxy (nginx/Caddy) with TLS termination. For anything beyond a
demo/portfolio deployment, swap:
- SQLite -> PostgreSQL (`DATABASE_URL`)
- The in-memory rate limiter -> Redis-backed limiting
- Local file uploads -> S3-compatible object storage

## 15. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` on startup | Activate your venv, re-run `pip install -r requirements.txt` |
| Registration fails with a bcrypt error | Ensure `bcrypt==4.0.1` is installed (newer bcrypt breaks passlib 1.7.4's backend detection) |
| Chat says "not configured" | Add at least one provider API key to `.env` and restart |
| Gmail/Calendar calls return 400 | Complete the OAuth flow from **Settings -> Connect Gmail/Calendar** first |
| MCP client hangs or errors | Never log to stdout inside `mcp_server.py` - stdout is reserved for protocol messages |
| Port already in use | Change `APP_PORT` in `.env`, or stop the other process |

## 16. Future improvements

- Swap the local hashing-embedding fallback for a small local sentence-transformer model
- Add WebSocket-based full-duplex voice streaming
- Add per-user rate limiting and usage dashboards
- Add PostgreSQL + pgvector for production-scale RAG
- Add role-based access control for multi-tenant deployments

---

See also: [`ARCHITECTURE.md`](ARCHITECTURE.md), [`API_REFERENCE.md`](API_REFERENCE.md),
[`ROADMAP.md`](ROADMAP.md), [`CONTRIBUTING.md`](CONTRIBUTING.md),
[`SECURITY.md`](SECURITY.md), [`CHANGELOG.md`](CHANGELOG.md).
