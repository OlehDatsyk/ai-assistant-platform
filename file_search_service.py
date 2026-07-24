"""
file_search_service.py
Search across the user's uploaded document library by metadata (filename,
type) and by semantic content (delegates to rag_service's vector search).
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import rag_service
from models import Document


async def search_by_metadata(db: AsyncSession, user_id: str, query: str) -> list[Document]:
    stmt = select(Document).where(
        Document.user_id == user_id, Document.filename.ilike(f"%{query}%")
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def search_by_content(db: AsyncSession, user_id: str, query: str, top_k: int = 10) -> list[dict]:
    # Restrict semantic search to this user's documents.
    stmt = select(Document.id).where(Document.user_id == user_id)
    result = await db.execute(stmt)
    doc_ids = [row[0] for row in result.all()]
    if not doc_ids:
        return []
    return await rag_service.semantic_search(db, query, doc_ids, top_k=top_k)
