"""routes_settings.py — user preference endpoints (default model, theme, system prompt)."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import get_current_user
from database import get_db
from models import User, UserPreference

router = APIRouter(prefix="/api/settings", tags=["settings"])


class PreferencesUpdate(BaseModel):
    default_model: str | None = None
    theme: str | None = None
    default_system_prompt: str | None = None


@router.get("")
async def get_preferences(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(UserPreference).where(UserPreference.user_id == user.id))
    prefs = result.scalar_one_or_none()
    if prefs is None:
        prefs = UserPreference(user_id=user.id)
        db.add(prefs)
        await db.commit()
        await db.refresh(prefs)
    return {
        "default_model": prefs.default_model,
        "theme": prefs.theme,
        "default_system_prompt": prefs.default_system_prompt,
    }


@router.put("")
async def update_preferences(
    payload: PreferencesUpdate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(UserPreference).where(UserPreference.user_id == user.id))
    prefs = result.scalar_one_or_none()
    if prefs is None:
        prefs = UserPreference(user_id=user.id)
        db.add(prefs)

    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(prefs, field, value)

    await db.commit()
    await db.refresh(prefs)
    return {
        "default_model": prefs.default_model,
        "theme": prefs.theme,
        "default_system_prompt": prefs.default_system_prompt,
    }
