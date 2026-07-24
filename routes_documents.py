"""routes_documents.py — document upload, library listing, and RAG query endpoints."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import rag_service
from auth import get_current_user
from config import UPLOADS_DIR
from database import AsyncSessionLocal, get_db
from models import Document, User
from schemas import DocumentOut, RagQuery

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_TYPES = {"pdf", "txt", "md"}


async def _ingest_in_background(document_id: str) -> None:
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Document).where(Document.id == document_id))
        document = result.scalar_one_or_none()
        if document:
            await rag_service.ingest_document(db, document)


@router.post("/upload", response_model=DocumentOut)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ext = (file.filename.rsplit(".", 1)[-1] if "." in file.filename else "").lower()
    if ext not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_TYPES)}")

    content = await file.read()
    document = Document(user_id=user.id, filename=file.filename, file_type=ext, size_bytes=len(content))
    db.add(document)
    await db.commit()
    await db.refresh(document)

    dest = Path(UPLOADS_DIR) / f"{document.id}_{document.filename}"
    dest.write_bytes(content)

    background_tasks.add_task(_ingest_in_background, document.id)
    return document


@router.get("", response_model=list[DocumentOut])
async def list_documents(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.user_id == user.id).order_by(Document.created_at.desc()))
    return result.scalars().all()


@router.delete("/{document_id}")
async def delete_document(
    document_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if document is None or document.user_id != user.id:
        raise HTTPException(404, "Document not found")
    file_path = Path(UPLOADS_DIR) / f"{document.id}_{document.filename}"
    file_path.unlink(missing_ok=True)
    await db.delete(document)
    await db.commit()
    return {"status": "deleted"}


@router.post("/query")
async def query_documents(
    payload: RagQuery, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    results = await rag_service.semantic_search(db, payload.query, payload.document_ids or None, payload.top_k)
    return {"results": results}
