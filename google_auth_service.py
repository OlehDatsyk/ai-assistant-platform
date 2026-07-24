"""
google_auth_service.py
Shared Google OAuth2 flow for Gmail + Calendar integrations. Requires the user
to set GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET in .env and complete the OAuth
consent flow once; the resulting token is stored in IntegrationToken.
"""
from __future__ import annotations

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models import IntegrationToken

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
]


def build_flow() -> Flow:
    client_config = {
        "web": {
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [settings.google_redirect_uri],
        }
    }
    return Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=settings.google_redirect_uri)


def get_authorization_url() -> str:
    flow = build_flow()
    auth_url, _ = flow.authorization_url(access_type="offline", prompt="consent")
    return auth_url


async def exchange_code_and_store(db: AsyncSession, user_id: str, code: str) -> IntegrationToken:
    flow = build_flow()
    flow.fetch_token(code=code)
    creds = flow.credentials

    result = await db.execute(
        select(IntegrationToken).where(
            IntegrationToken.user_id == user_id, IntegrationToken.provider == "google"
        )
    )
    token_row = result.scalar_one_or_none()
    if token_row is None:
        token_row = IntegrationToken(user_id=user_id, provider="google")
        db.add(token_row)

    token_row.access_token = creds.token or ""
    token_row.refresh_token = creds.refresh_token or token_row.refresh_token or ""
    token_row.extra = {"token_uri": creds.token_uri, "scopes": creds.scopes}
    await db.commit()
    await db.refresh(token_row)
    return token_row


async def get_credentials(db: AsyncSession, user_id: str) -> Credentials | None:
    result = await db.execute(
        select(IntegrationToken).where(
            IntegrationToken.user_id == user_id, IntegrationToken.provider == "google"
        )
    )
    row = result.scalar_one_or_none()
    if row is None or not row.access_token:
        return None
    return Credentials(
        token=row.access_token,
        refresh_token=row.refresh_token,
        token_uri=(row.extra or {}).get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        scopes=(row.extra or {}).get("scopes", SCOPES),
    )
