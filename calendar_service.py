"""
calendar_service.py
Google Calendar integration: list, create, update, delete events, plus an
AI scheduling assistant that turns natural language into event proposals.
"""
from __future__ import annotations

import json

from googleapiclient.discovery import build
from sqlalchemy.ext.asyncio import AsyncSession

import google_auth_service
from llm_providers import ChatMessage, get_provider
from logging_config import get_logger

logger = get_logger(__name__)


async def _calendar_client(db: AsyncSession, user_id: str):
    creds = await google_auth_service.get_credentials(db, user_id)
    if creds is None:
        raise RuntimeError("Calendar is not connected for this user. Complete the OAuth flow first.")
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


async def list_events(db: AsyncSession, user_id: str, max_results: int = 20) -> list[dict]:
    service = await _calendar_client(db, user_id)
    resp = service.events().list(
        calendarId="primary", maxResults=max_results, singleEvents=True, orderBy="startTime"
    ).execute()
    return resp.get("items", [])


async def create_event(
    db: AsyncSession, user_id: str, summary: str, description: str, start_iso: str, end_iso: str,
    attendees: list[str] | None = None,
) -> dict:
    service = await _calendar_client(db, user_id)
    body = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start_iso},
        "end": {"dateTime": end_iso},
        "attendees": [{"email": a} for a in (attendees or [])],
    }
    return service.events().insert(calendarId="primary", body=body).execute()


async def update_event(db: AsyncSession, user_id: str, event_id: str, patch: dict) -> dict:
    service = await _calendar_client(db, user_id)
    return service.events().patch(calendarId="primary", eventId=event_id, body=patch).execute()


async def delete_event(db: AsyncSession, user_id: str, event_id: str) -> dict:
    service = await _calendar_client(db, user_id)
    service.events().delete(calendarId="primary", eventId=event_id).execute()
    return {"status": "deleted", "event_id": event_id}


async def ai_schedule_from_text(request_text: str, provider_name: str = "openai") -> dict:
    """Parse a natural-language scheduling request into a structured event proposal."""
    provider = get_provider(provider_name)
    prompt = (
        "Extract a calendar event from this request. Respond ONLY with JSON: "
        '{"summary": str, "description": str, "start_iso": str, "end_iso": str, "attendees": [str]}. '
        "Use ISO 8601 datetimes. If a time isn't specified, make a reasonable assumption.\n\n"
        f"Request: {request_text}"
    )
    result = await provider.complete([ChatMessage(role="user", content=prompt)])
    text = result.text.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Scheduling assistant returned non-JSON: %s", text)
        return {"error": "Could not parse a structured event from that request.", "raw": text}
