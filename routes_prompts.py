"""routes_prompts.py - prompt library CRUD endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import get_current_user
from database import get_db
from models import PromptTemplate, User
from schemas import PromptCreate, PromptOut

router = APIRouter(prefix="/api/prompts", tags=["prompts"])


@router.get("", response_model=list[PromptOut])
async def list_prompts(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(PromptTemplate).where(PromptTemplate.user_id == user.id).order_by(PromptTemplate.created_at.desc())
    )
    return result.scalars().all()


@router.post("", response_model=PromptOut)
async def create_prompt(
    payload: PromptCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    prompt = PromptTemplate(user_id=user.id, title=payload.title, content=payload.content, category=payload.category)
    db.add(prompt)
    await db.commit()
    await db.refresh(prompt)
    return prompt


@router.patch("/{prompt_id}/favorite", response_model=PromptOut)
async def toggle_favorite(
    prompt_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(PromptTemplate).where(PromptTemplate.id == prompt_id))
    prompt = result.scalar_one_or_none()
    if prompt is None or prompt.user_id != user.id:
        raise HTTPException(404, "Prompt not found")
    prompt.is_favorite = not prompt.is_favorite
    await db.commit()
    await db.refresh(prompt)
    return prompt


@router.delete("/{prompt_id}")
async def delete_prompt(
    prompt_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(PromptTemplate).where(PromptTemplate.id == prompt_id))
    prompt = result.scalar_one_or_none()
    if prompt is None or prompt.user_id != user.id:
        raise HTTPException(404, "Prompt not found")
    await db.delete(prompt)
    await db.commit()
    return {"status": "deleted"}
