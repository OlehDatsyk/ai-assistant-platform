# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - Initial release

### Added
- Multi-model chat (OpenAI, Anthropic Claude, Google Gemini) with streaming responses
- Multi-conversation support, conversation history, pinning
- Short-term (windowed) and long-term (persisted, auto-extracted) memory
- Prompt library with favorites
- Retrieval-augmented generation: PDF/TXT/MD upload, chunking, embeddings, semantic
  search with source citations
- Tool / function calling with a shared tool registry (web search, calculator)
- Code assistant: generate, explain, debug, refactor, review, generate tests
- Multi-step agent framework with research / coding / email personas
- Full MCP (Model Context Protocol) server and client demonstrating tools, resources,
  and prompts
- Google Calendar integration with an AI scheduling assistant
- Gmail integration: read, search, summarize, draft, send
- Telegram bot mirroring the web chat pipeline
- Voice: speech-to-text (Whisper) and text-to-speech (OpenAI TTS)
- Vision: OCR, image captioning, receipt analysis
- File search across the document library (metadata + semantic)
- Outbound automation webhooks for n8n / Zapier / Make
- JWT authentication, bcrypt password hashing, per-IP rate limiting
- Dark-mode responsive dashboard UI (no frontend build step)
- Pytest suite covering auth, CRUD flows, RAG math, and tool-calling safety
- Cross-platform one-click startup scripts (Windows `.bat`, macOS `.command`)

### Fixed (found via live end-to-end testing during development)
- `bcrypt`/`passlib` incompatibility that broke password hashing on registration
  (pinned `bcrypt==4.0.1`)
- MCP server logging to stdout, which corrupted the stdio JSON-RPC protocol stream
  (moved to a stderr-only logger)
- MCP resource lookup failing due to an `AnyUrl` vs `str` type mismatch
- MCP prompt handler returning a bare list instead of the expected `GetPromptResult`
