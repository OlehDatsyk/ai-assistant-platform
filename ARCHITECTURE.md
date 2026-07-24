# Architecture

## Design goals

1. **Flat and readable** - nearly all modules live in the repository root; folders are
   only used where the framework requires it (`static/`, `templates/`) or where grouping
   clearly aids navigation (`tests/`, `docs/`).
2. **Provider-agnostic core** - the chat, RAG, agent, and code-assistant layers never talk
   to OpenAI/Anthropic/Gemini SDKs directly; they go through `llm_providers.get_provider()`.
3. **Service layer between routes and data** - FastAPI routers (`routes_*.py`) stay thin;
   business logic lives in `*_service.py` modules so it's reusable from the Telegram bot,
   the MCP server, and pytest without duplicating logic.

## Request lifecycle (chat example)

```
Browser
  │ POST /api/chat/conversations/{id}/stream
  ▼
routes_chat.stream_message()
  │  - resolves the owned Conversation
  │  - delegates to chat_service.generate_reply_stream()
  ▼
chat_service.generate_reply_stream()
  │  - memory_service: pulls short-term window + long-term facts
  │  - rag_service (optional): semantic search + citation block
  │  - tools.run_tool_loop (optional): web search / calculator round-trips
  │  - llm_providers.get_provider(name).stream_chat(): streams tokens
  │  - persists both User and Assistant Message rows
  ▼
StreamingResponse (text/event-stream) -> browser renders tokens live
```

## Data model

See `models.py` for the full SQLAlchemy schema. Key relationships:

```
User 1--1 UserPreference
User 1--* Conversation 1--* Message
User 1--* MemoryItem
User 1--* PromptTemplate
User 1--* Document 1--* DocumentChunk
User 1--* IntegrationToken   (Google OAuth, Telegram, ...)
```

## Multi-model abstraction

`llm_providers.py` defines a small `LLMProvider` ABC with two methods:
- `stream_chat(messages, tools, temperature) -> AsyncIterator[str]`
- `complete(messages, tools, temperature) -> CompletionResult` (used for tool-calling
  round trips and structured/JSON extraction, where streaming isn't useful)

Each concrete provider (`OpenAIProvider`, `AnthropicProvider`, `GeminiProvider`) adapts
the shared `ChatMessage`/`ToolSpec` types to that vendor's SDK shape. Adding a fourth
provider means implementing this one interface - nothing else in the app changes.

## RAG pipeline

```
Upload -> extract_text() -> chunk_text() -> embed_texts() -> DocumentChunk rows (SQLite)
                                                                  |
Query  ----------------------------------------------------------+
  │ embed_texts([query])
  ▼
cosine_similarity() against every candidate chunk -> top-K -> format_rag_context()
  -> injected into the system prompt as "[Source N]" blocks
```

Embeddings default to OpenAI's `text-embedding-3-small` when a key is present, and fall
back to a deterministic local hashing embedding otherwise - this keeps the RAG feature
usable in a zero-API-key demo, at reduced retrieval quality.

## Memory

- **Short-term** = last N messages of the *current* conversation (windowed, verbatim).
- **Long-term** = durable facts extracted heuristically + via a cheap LLM call after each
  user message, persisted per-user (not per-conversation), and injected into *every*
  conversation's system prompt. This is what lets the assistant "remember" a user's stated
  preferences across separate chats.

## Tool / function calling

`tools.py` holds a single source of truth: a list of `ToolSpec` (JSON-schema definitions)
and a matching `TOOL_IMPLEMENTATIONS` dict. This same registry is used by:
- the in-app chat tool-calling loop (`chat_service.run_tool_loop`)
- the agent loop (`agents.run_agent`)
- the MCP server (`mcp_server.py`), so external MCP clients get the identical tool behavior

## Agents

`agents.py` implements a bounded plan-act-observe loop (max 6 steps) around the same
tool-calling primitive used by chat. Three personas (research / coding / email) are just
different system prompts over the same loop - a deliberate choice to keep the agent
runtime itself simple and auditable rather than building a heavier framework.

## MCP server/client

Implemented directly against the official `mcp` Python SDK's stdio transport. Important
operational detail: **stdout is reserved for the JSON-RPC protocol stream**, so
`mcp_server.py` configures its own stderr-only logger rather than using the shared
`logging_config` module (which logs to stdout for the web app, where that's fine).

## Security layers

- Passwords hashed with bcrypt via passlib (pinned to a known-compatible bcrypt version)
- JWT bearer auth (`auth.py`), verified per-request via a FastAPI dependency
- Per-IP sliding-window rate limiting (`rate_limiter.py`) - swap for Redis in production
- Calculator tool uses an AST-based safe evaluator, never `eval()`
- All user-facing routes scope queries by `user_id` to prevent cross-tenant access
