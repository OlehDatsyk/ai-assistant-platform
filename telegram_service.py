"""
telegram_service.py
Telegram bot that mirrors the platform's chat: every message from a Telegram
user is routed through the same chat_service used by the web UI, and replies
are streamed back to Telegram. Run directly:

    python telegram_service.py

Requires TELEGRAM_BOT_TOKEN in .env.
"""
from __future__ import annotations

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from config import settings
from database import AsyncSessionLocal
from llm_providers import ChatMessage, get_provider
from logging_config import configure_logging, get_logger
from models import Conversation, User

configure_logging()
logger = get_logger(__name__)

DEFAULT_PROVIDER = "openai"


async def _get_or_create_telegram_user(db, chat_id: str) -> User:
    """Maps a Telegram chat to a lightweight platform user, creating one on first contact."""
    from sqlalchemy import select

    email = f"telegram_{chat_id}@bot.local"
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(email=email, hashed_password="", full_name=f"Telegram {chat_id}")
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return user


async def _get_or_create_conversation(db, user: User) -> Conversation:
    from sqlalchemy import select

    result = await db.execute(
        select(Conversation).where(Conversation.user_id == user.id).order_by(Conversation.updated_at.desc())
    )
    conv = result.scalars().first()
    if conv is None:
        conv = Conversation(user_id=user.id, title="Telegram chat", model_provider=DEFAULT_PROVIDER)
        db.add(conv)
        await db.commit()
        await db.refresh(conv)
    return conv


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text
    chat_id = str(update.effective_chat.id)

    async with AsyncSessionLocal() as db:
        user = await _get_or_create_telegram_user(db, chat_id)
        conversation = await _get_or_create_conversation(db, user)

        provider = get_provider(conversation.model_provider)
        if not provider.is_configured():
            await update.message.reply_text(
                "The configured AI provider is missing an API key. Please contact the admin."
            )
            return

        messages = [
            ChatMessage(role="system", content=conversation.system_prompt),
            ChatMessage(role="user", content=text),
        ]
        result = await provider.complete(messages)
        await update.message.reply_text(result.text or "(empty response)")

        from models import Message

        db.add(Message(conversation_id=conversation.id, role="user", content=text))
        db.add(Message(conversation_id=conversation.id, role="assistant", content=result.text or ""))
        await db.commit()


def build_application() -> Application:
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")
    application = Application.builder().token(settings.telegram_bot_token).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    return application


if __name__ == "__main__":
    app = build_application()
    logger.info("Starting Telegram bot polling loop...")
    app.run_polling()
