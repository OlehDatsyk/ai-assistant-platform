"""
gmail_service.py
Gmail integration: read, search, summarize, draft, and send emails via the
Gmail API using the credentials obtained through google_auth_service.
"""
from __future__ import annotations

import base64
from email.mime.text import MIMEText

from googleapiclient.discovery import build
from sqlalchemy.ext.asyncio import AsyncSession

import google_auth_service
from llm_providers import ChatMessage, get_provider
from logging_config import get_logger

logger = get_logger(__name__)


async def _gmail_client(db: AsyncSession, user_id: str):
    creds = await google_auth_service.get_credentials(db, user_id)
    if creds is None:
        raise RuntimeError("Gmail is not connected for this user. Complete the OAuth flow first.")
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


async def list_recent_emails(db: AsyncSession, user_id: str, max_results: int = 10) -> list[dict]:
    service = await _gmail_client(db, user_id)
    resp = service.users().messages().list(userId="me", maxResults=max_results).execute()
    messages = []
    for m in resp.get("messages", []):
        full = service.users().messages().get(userId="me", id=m["id"], format="metadata",
                                                metadataHeaders=["Subject", "From", "Date"]).execute()
        headers = {h["name"]: h["value"] for h in full.get("payload", {}).get("headers", [])}
        messages.append(
            {
                "id": m["id"],
                "subject": headers.get("Subject", "(no subject)"),
                "from": headers.get("From", ""),
                "date": headers.get("Date", ""),
                "snippet": full.get("snippet", ""),
            }
        )
    return messages


async def search_emails(db: AsyncSession, user_id: str, query: str, max_results: int = 10) -> list[dict]:
    service = await _gmail_client(db, user_id)
    resp = service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
    return [{"id": m["id"]} for m in resp.get("messages", [])] 


async def summarize_inbox(db: AsyncSession, user_id: str, provider_name: str = "openai") -> str:
    emails = await list_recent_emails(db, user_id, max_results=10)
    if not emails:
        return "No recent emails found."
    listing = "\n".join(f"- From {e['from']}: {e['subject']} — {e['snippet']}" for e in emails)
    provider = get_provider(provider_name)
    result = await provider.complete(
        [
            ChatMessage(role="system", content="Summarize this inbox into priorities and action items."),
            ChatMessage(role="user", content=listing),
        ]
    )
    return result.text


async def draft_reply(
    db: AsyncSession, user_id: str, original_message: str, intent: str, provider_name: str = "openai"
) -> str:
    provider = get_provider(provider_name)
    result = await provider.complete(
        [
            ChatMessage(role="system", content="Draft a concise, professional email reply."),
            ChatMessage(
                role="user",
                content=f"Original message:\n{original_message}\n\nReply intent: {intent}",
            ),
        ]
    )
    return result.text


async def send_email(db: AsyncSession, user_id: str, to: str, subject: str, body: str) -> dict:
    service = await _gmail_client(db, user_id)
    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
    return {"id": sent.get("id"), "status": "sent"}
