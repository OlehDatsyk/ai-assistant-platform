"""
config.py
Centralized application configuration using pydantic-settings.
All environment variables are loaded once here and imported elsewhere.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
VECTOR_DIR = DATA_DIR / "vector_store"
LOG_DIR = BASE_DIR / "logs"

for directory in (DATA_DIR, UPLOADS_DIR, VECTOR_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Strongly typed application settings, populated from environment / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Core
    app_name: str = "AI Assistant Platform"
    app_env: str = "development"
    app_secret_key: str = "insecure-dev-key"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    database_url: str = f"sqlite+aiosqlite:///{DATA_DIR / 'app.db'}"

    # Auth
    jwt_secret_key: str = "insecure-jwt-key"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    # Providers
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    google_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # Web search
    tavily_api_key: str = ""

    # Google OAuth (Gmail / Calendar)
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/api/gmail/oauth/callback"

    # Telegram
    telegram_bot_token: str = ""

    # Voice
    openai_tts_model: str = "tts-1"
    openai_stt_model: str = "whisper-1"

    # Automation
    n8n_webhook_url: str = ""
    zapier_webhook_url: str = ""
    make_webhook_url: str = ""

    # Rate limiting
    rate_limit_per_minute: int = 60

    @property
    def providers_available(self) -> dict[str, bool]:
        return {
            "openai": bool(self.openai_api_key),
            "anthropic": bool(self.anthropic_api_key),
            "gemini": bool(self.google_api_key),
        }


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor so we parse the environment only once."""
    return Settings()


settings = get_settings()
