# Roadmap

## Near-term
- [ ] Swap the local hashing-embedding RAG fallback for a small local sentence-transformer
      model so offline retrieval quality improves without needing an API key
- [ ] WebSocket-based full-duplex voice streaming (currently record-then-send)
- [ ] Structured output mode exposed directly in the chat UI (not just via API)
- [ ] Per-conversation tool enable/disable toggles in the UI

## Mid-term
- [ ] PostgreSQL + pgvector support as a drop-in alternative to SQLite for production RAG
- [ ] Redis-backed rate limiting and session cache for multi-worker deployments
- [ ] Role-based access control (admin / member) for multi-tenant deployments
- [ ] Usage dashboards (tokens, cost estimates per provider, per user)
- [ ] Richer agent framework: parallel tool calls, sub-agent delegation

## Long-term
- [ ] Plugin system for third-party tool packs (beyond the built-in registry)
- [ ] Native mobile client
- [ ] Fine-tuned routing: automatically pick the cheapest/best model per request type
- [ ] Team workspaces with shared prompt libraries and shared documents

Have an idea? Open an issue or see [`CONTRIBUTING.md`](CONTRIBUTING.md).
