"""routes_media_misc.py — voice, vision, file search, and automation webhook endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

import automation_service
import file_search_service
import vision_service
import voice_service
from auth import get_current_user
from database import get_db
from models import User

router = APIRouter(prefix="/api", tags=["media-misc"])


# --------------------------------------------------------------- Voice ------
@router.post("/voice/transcribe")
async def transcribe(file: UploadFile, user: User = Depends(get_current_user)):
    audio_bytes = await file.read()
    try:
        text = await voice_service.transcribe_audio(audio_bytes, file.filename)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"text": text}


@router.post("/voice/speak")
async def speak(text: str, voice: str = "alloy", user: User = Depends(get_current_user)):
    try:
        audio = await voice_service.synthesize_speech(text, voice)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    return Response(content=audio, media_type="audio/mpeg")


# -------------------------------------------------------------- Vision ------
@router.post("/vision/ocr")
async def ocr(file: UploadFile, user: User = Depends(get_current_user)):
    image_bytes = await file.read()
    try:
        return {"text": await vision_service.ocr_image(image_bytes)}
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/vision/caption")
async def caption(file: UploadFile, user: User = Depends(get_current_user)):
    image_bytes = await file.read()
    try:
        return {"caption": await vision_service.caption_image(image_bytes)}
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/vision/receipt")
async def receipt(file: UploadFile, user: User = Depends(get_current_user)):
    image_bytes = await file.read()
    try:
        return await vision_service.analyze_receipt(image_bytes)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc


# ---------------------------------------------------------- File search -----
@router.get("/files/search")
async def search_files(
    q: str = Query(...), mode: str = Query("content"), user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    if mode == "metadata":
        return await file_search_service.search_by_metadata(db, user.id, q)
    return await file_search_service.search_by_content(db, user.id, q)


# ---------------------------------------------------------- Automation ------
@router.post("/automation/{platform}/trigger")
async def trigger_automation(platform: str, event: str, payload: dict, user: User = Depends(get_current_user)):
    return await automation_service.trigger_webhook(platform, event, payload)
