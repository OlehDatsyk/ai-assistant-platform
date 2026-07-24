# Contributing

Thanks for your interest in improving the AI Assistant Platform.

## Getting set up
1. Fork and clone the repository.
2. Follow the installation steps in [`README.md`](README.md).
3. Run the test suite before making changes, so you have a known-good baseline:
   ```bash
   pytest
   ```

## Development guidelines
- **Type hints everywhere.** Every function signature should be fully typed.
- **Async first.** Route handlers and service functions that touch I/O (DB, HTTP, LLM
  calls) should be `async def`.
- **Keep the service layer thin and reusable.** Business logic belongs in `*_service.py`
  modules, not inside `routes_*.py` handlers - the same logic is reused by the Telegram
  bot and the MCP server.
- **One provider, one adapter.** If you add a new LLM vendor, implement the `LLMProvider`
  ABC in `llm_providers.py` rather than special-casing it elsewhere.
- **No secrets in code.** All credentials come from `.env` via `config.py`.
- **Write a test with every behavioral change.** See `tests/` for patterns - API tests use
  the async httpx client against an in-memory SQLite DB; unit tests import modules directly.

## Submitting changes
1. Create a feature branch: `git checkout -b feature/my-change`
2. Make your changes with clear, atomic commits.
3. Run `pytest` and ensure everything passes.
4. Open a pull request describing what changed and why.

## Reporting bugs
Open an issue with:
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Relevant log output (from `logs/app.log` or the terminal)

## Code of conduct
Participation in this project is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
