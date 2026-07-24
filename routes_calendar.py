"""routes_calendar.py — Google Calendar CRUD + AI scheduling assistant."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

import calendar_service
from auth import get_current_user
from database import get_db
from models import User
from schemas import EventCreate

router = APIRouter(prefix="/api/calendar", tags=["calendar"])


@router.get("/events")
async def get_events(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    try:
        return await calendar_service.list_events(db, user.id)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/events")
async def create_event(
    payload: EventCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    try:
        return await calendar_service.create_event(
            db, user.id, payload.summary, payload.description, payload.start_iso, payload.end_iso, payload.attendees
        )
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.patch("/events/{event_id}")
async def update_event(
    event_id: str, patch: dict, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    try:
        return await calendar_service.update_event(db, user.id, event_id, patch)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/events/{event_id}")
async def delete_event(
    event_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    try:
        return await calendar_service.delete_event(db, user.id, event_id)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/ai-schedule")
async def ai_schedule(text: str = Query(...), user: User = Depends(get_current_user)):
    return await calendar_service.ai_schedule_from_text(text)
