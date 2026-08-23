"""routes_memory.py - long-term memory CRUD endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import get_current_user
from database import get_db
from models import MemoryItem, User
from schemas import MemoryCreate, MemoryOut

router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.get("", response_model=list[MemoryOut])
async def list_memory(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(MemoryItem).where(MemoryItem.user_id == user.id).order_by(MemoryItem.created_at.desc())
    )
    return result.scalars().all()


@router.post("", response_model=MemoryOut)
async def add_memory(
    payload: MemoryCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    item = MemoryItem(user_id=user.id, content=payload.content, category=payload.category, importance=payload.importance)
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return item


@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(MemoryItem).where(MemoryItem.id == memory_id))
    item = result.scalar_one_or_none()
    if item is None or item.user_id != user.id:
        raise HTTPException(404, "Memory item not found")
    await db.delete(item)
    await db.commit()
    return {"status": "deleted"}
