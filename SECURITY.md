# Security Policy

## Reporting a vulnerability
If you discover a security vulnerability in this project, please report it privately
rather than opening a public issue. Include:
- A description of the vulnerability and its potential impact
- Steps to reproduce
- Any suggested remediation

We aim to acknowledge reports within a few business days.

## Supported versions
Only the latest version on the `main` branch receives security fixes.

## Security practices in this codebase
- **Secrets management:** all API keys and secrets are loaded from `.env` (never
  committed - see `.gitignore`) via `config.py`. Nothing is hardcoded.
- **Password storage:** bcrypt via passlib; plaintext passwords are never logged or stored.
- **Auth:** stateless JWT bearer tokens with configurable expiry (`JWT_EXPIRE_MINUTES`).
- **Rate limiting:** a per-IP sliding-window limiter (`rate_limiter.py`) guards against
  basic abuse; swap for a Redis-backed limiter behind a load balancer.
- **Input validation:** all request bodies are validated with Pydantic v2 models
  (`schemas.py`) before touching business logic.
- **Tool-calling safety:** the `calculator` tool uses an AST-based safe evaluator - it
  never calls `eval()` or `exec()` on user input.
- **Tenant isolation:** every user-scoped query filters by `user_id`; ownership is
  re-checked on every read/update/delete of conversations, memory, prompts, and documents.
- **File uploads:** restricted to `.pdf`, `.txt`, `.md`; stored outside any web-served
  static directory.

## Hardening checklist before production use
- [ ] Replace `APP_SECRET_KEY` and `JWT_SECRET_KEY` with long random values
- [ ] Set `APP_ENV=production` (tightens CORS to an explicit origin list)
- [ ] Put the app behind HTTPS (reverse proxy with TLS termination)
- [ ] Move rate limiting to a shared store (Redis) if running multiple workers
- [ ] Rotate any API keys that were used during development/testing
- [ ] Review `google_client_secret` / OAuth redirect URIs for your production domain
