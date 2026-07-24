"""routes_chat.py — conversation CRUD and streaming chat endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import chat_service
from auth import get_current_user
from database import get_db
from models import Conversation, Message, User
from schemas import ConversationCreate, ConversationOut, MessageCreate, MessageOut

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/conversations", response_model=ConversationOut)
async def create_conversation(
    payload: ConversationCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Conversation:
    conv = Conversation(
        user_id=user.id,
        title=payload.title,
        model_provider=payload.model_provider,
        system_prompt=payload.system_prompt,
    )
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.is_pinned.desc(), Conversation.updated_at.desc())
    )
    return result.scalars().all()


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
async def get_messages(
    conversation_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    conv = await _get_owned_conversation(db, conversation_id, user.id)
    result = await db.execute(
        select(Message).where(Message.conversation_id == conv.id).order_by(Message.created_at)
    )
    return result.scalars().all()


@router.patch("/conversations/{conversation_id}/pin", response_model=ConversationOut)
async def toggle_pin(
    conversation_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    conv = await _get_owned_conversation(db, conversation_id, user.id)
    conv.is_pinned = not conv.is_pinned
    await db.commit()
    await db.refresh(conv)
    return conv


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    conv = await _get_owned_conversation(db, conversation_id, user.id)
    await db.delete(conv)
    await db.commit()
    return {"status": "deleted"}


@router.post("/conversations/{conversation_id}/stream")
async def stream_message(
    conversation_id: str,
    payload: MessageCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    conv = await _get_owned_conversation(db, conversation_id, user.id)
    provider_name = payload.model_provider or conv.model_provider

    async def event_stream():
        async for delta in chat_service.generate_reply_stream(
            db=db,
            conversation=conv,
            user_text=payload.content,
            provider_name=provider_name,
            use_rag=payload.use_rag,
            document_ids=payload.document_ids,
            use_web_search=payload.use_web_search,
        ):
            yield f"data: {delta}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def _get_owned_conversation(db: AsyncSession, conversation_id: str, user_id: str) -> Conversation:
    result = await db.execute(select(Conversation).where(Conversation.id == conversation_id))
    conv = result.scalar_one_or_none()
    if conv is None or conv.user_id != user_id:
        raise HTTPException(404, "Conversation not found")
    return conv
