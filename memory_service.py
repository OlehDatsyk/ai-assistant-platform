"""
memory_service.py
Short-term memory = recent conversation turns (windowed).
Long-term memory = durable facts/preferences stored in MemoryItem and
injected into the system prompt on every request.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_providers import ChatMessage, get_provider
from logging_config import get_logger
from models import MemoryItem, Message

logger = get_logger(__name__)

SHORT_TERM_WINDOW = 20  # number of recent messages kept verbatim in context


async def get_short_term_context(db: AsyncSession, conversation_id: str) -> list[Message]:
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(SHORT_TERM_WINDOW)
    )
    return list(reversed(result.scalars().all()))


async def get_long_term_memory(db: AsyncSession, user_id: str, limit: int = 15) -> list[MemoryItem]:
    result = await db.execute(
        select(MemoryItem)
        .where(MemoryItem.user_id == user_id)
        .order_by(MemoryItem.importance.desc(), MemoryItem.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


def format_memory_block(items: list[MemoryItem]) -> str:
    if not items:
        return ""
    lines = "\n".join(f"- ({m.category}) {m.content}" for m in items)
    return f"\n\n# Long-term memory about this user\n{lines}\n"


async def maybe_extract_memory(
    db: AsyncSession, user_id: str, user_message: str, provider_name: str = "openai"
) -> MemoryItem | None:
    """
    Heuristic + LLM-assisted extraction: ask the model whether this message contains
    a durable fact worth remembering (name, preference, recurring constraint, etc).
    Cheap guard first so we don't spend a call on every trivial message.
    """
    trivial_markers = ("hi", "hello", "thanks", "ok", "yes", "no")
    if len(user_message.strip()) < 15 or user_message.strip().lower() in trivial_markers:
        return None

    try:
        provider = get_provider(provider_name)
        if not provider.is_configured():
            return None
        prompt = (
            "Decide if this message contains a durable personal fact, preference, or "
            "constraint worth remembering long-term (e.g. name, role, tech stack, timezone, "
            "recurring preference). If yes, respond with just the fact in one short sentence. "
            "If no, respond with exactly: NONE.\n\nMessage: " + user_message
        )
        result = await provider.complete([ChatMessage(role="user", content=prompt)])
        text = result.text.strip()
        if not text or text.upper().startswith("NONE"):
            return None
        item = MemoryItem(user_id=user_id, content=text, category="auto", importance=2)
        db.add(item)
        await db.commit()
        await db.refresh(item)
        return item
    except Exception:  # noqa: BLE001
        logger.exception("Memory extraction failed; continuing without it")
        return None
