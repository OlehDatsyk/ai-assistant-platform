"""
automation_service.py
Outbound automation integrations. Rather than reimplementing n8n/Zapier/Make,
the platform exposes a generic webhook dispatcher: any event in the app
(new conversation, document ingested, agent finished, etc.) can be POSTed to a
configured automation platform's webhook URL, which then triggers whatever
workflow the user has built there.
"""
from __future__ import annotations

from typing import Any

import httpx

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)

_TARGETS = {
    "n8n": lambda: settings.n8n_webhook_url,
    "zapier": lambda: settings.zapier_webhook_url,
    "make": lambda: settings.make_webhook_url,
}


async def trigger_webhook(platform: str, event: str, payload: dict[str, Any]) -> dict[str, Any]:
    getter = _TARGETS.get(platform)
    if getter is None:
        return {"error": f"Unknown automation platform: {platform}"}
    url = getter()
    if not url:
        return {"error": f"No webhook URL configured for {platform}. Set it in .env."}

    body = {"event": event, "payload": payload}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
        return {"status": "sent", "platform": platform, "event": event}
    except httpx.HTTPError as exc:
        logger.exception("Automation webhook to %s failed", platform)
        return {"error": str(exc)}


async def broadcast_event(event: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Send the same event to every configured automation platform."""
    results = []
    for platform in _TARGETS:
        results.append(await trigger_webhook(platform, event, payload))
    return results
