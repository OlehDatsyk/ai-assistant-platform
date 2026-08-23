"""routes_gmail.py - Gmail OAuth flow + read/summarize/draft/send/search endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

import gmail_service
import google_auth_service
from auth import get_current_user
from database import get_db
from models import User

router = APIRouter(prefix="/api/gmail", tags=["gmail"])


@router.get("/oauth/start")
async def oauth_start(user: User = Depends(get_current_user)):
    return {"authorization_url": google_auth_service.get_authorization_url()}


@router.get("/oauth/callback")
async def oauth_callback(code: str = Query(...), user_id: str = Query(...), db: AsyncSession = Depends(get_db)):
    await google_auth_service.exchange_code_and_store(db, user_id, code)
    return RedirectResponse(url="/settings?gmail_connected=1")


@router.get("/emails")
async def emails(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    try:
        return await gmail_service.list_recent_emails(db, user.id)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/emails/search")
async def search_emails(q: str = Query(...), user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    try:
        return await gmail_service.search_emails(db, user.id, q)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/summary")
async def summarize(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    try:
        return {"summary": await gmail_service.summarize_inbox(db, user.id)}
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/draft")
async def draft_reply(
    original_message: str, intent: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    return {"draft": await gmail_service.draft_reply(db, user.id, original_message, intent)}


@router.post("/send")
async def send_email(
    to: str, subject: str, body: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    try:
        return await gmail_service.send_email(db, user.id, to, subject, body)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
