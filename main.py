"""
main.py
FastAPI application entrypoint. Mounts all routers, static files, middleware,
and runs DB initialization on startup.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import BASE_DIR, settings
from database import init_db
from logging_config import configure_logging, get_logger
from rate_limiter import RateLimitMiddleware

import routes_auth
import routes_calendar
import routes_chat
import routes_code_agents
import routes_documents
import routes_gmail
import routes_media_misc
import routes_memory
import routes_pages
import routes_prompts
import routes_settings

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s (%s)", settings.app_name, settings.app_env)
    await init_db()
    configured = [k for k, v in settings.providers_available.items() if v]
    if configured:
        logger.info("Configured model providers: %s", ", ".join(configured))
    else:
        logger.warning("No model providers configured. Add API keys to .env to enable chat.")
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.app_env == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# API routers
app.include_router(routes_auth.router)
app.include_router(routes_chat.router)
app.include_router(routes_memory.router)
app.include_router(routes_prompts.router)
app.include_router(routes_documents.router)
app.include_router(routes_code_agents.router)
app.include_router(routes_calendar.router)
app.include_router(routes_gmail.router)
app.include_router(routes_media_misc.router)
app.include_router(routes_settings.router)

# Page (HTML) router — must be included after API routers
app.include_router(routes_pages.router)


@app.get("/health", tags=["system"])
async def health_check() -> dict:
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.app_host, port=settings.app_port, reload=True)
