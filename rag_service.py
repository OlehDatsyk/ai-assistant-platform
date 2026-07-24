"""
rag_service.py
Minimal, dependency-light RAG pipeline:
  - extract text from PDF / TXT / Markdown
  - chunk with overlap
  - embed via OpenAI embeddings (falls back to a local hashing embedding if
    no OpenAI key is configured, so the feature still works offline/demo-mode)
  - cosine-similarity search across stored chunks, returned with citations
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import UPLOADS_DIR, settings
from logging_config import get_logger
from models import Document, DocumentChunk

logger = get_logger(__name__)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
EMBED_DIM = 384


# ---------------------------------------------------------------- Extract ---
def extract_text(path: Path, file_type: str) -> str:
    if file_type == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


# ---------------------------------------------------------------- Chunk -----
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


# -------------------------------------------------------------- Embedding ---
def _hash_embedding(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Deterministic local fallback embedding (no external API needed)."""
    vec = np.zeros(dim, dtype=np.float32)
    for token in text.lower().split():
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    return (vec / norm if norm else vec).tolist()


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if settings.openai_api_key:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=settings.openai_api_key)
            resp = await client.embeddings.create(model="text-embedding-3-small", input=texts)
            return [d.embedding for d in resp.data]
        except Exception:  # noqa: BLE001
            logger.exception("OpenAI embedding failed, falling back to local hashing embedding")
    return [_hash_embedding(t) for t in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1e-9
    return float(np.dot(va, vb) / denom)


# ---------------------------------------------------------------- Pipeline --
async def ingest_document(db: AsyncSession, document: Document) -> None:
    """Full ingest pipeline for a previously-saved upload: extract -> chunk -> embed -> store."""
    path = UPLOADS_DIR / f"{document.id}_{document.filename}"
    try:
        text = extract_text(path, document.file_type)
        pieces = chunk_text(text)
        if not pieces:
            document.status = "empty"
            await db.commit()
            return
        embeddings = await embed_texts(pieces)
        for idx, (piece, emb) in enumerate(zip(pieces, embeddings)):
            db.add(
                DocumentChunk(document_id=document.id, chunk_index=idx, content=piece, embedding=emb)
            )
        document.status = "ready"
        await db.commit()
        logger.info("Ingested %s chunks for document %s", len(pieces), document.id)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to ingest document %s", document.id)
        document.status = "error"
        await db.commit()


async def semantic_search(
    db: AsyncSession, query: str, document_ids: list[str] | None, top_k: int = 5
) -> list[dict]:
    stmt = select(DocumentChunk)
    if document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
    result = await db.execute(stmt)
    chunks = result.scalars().all()
    if not chunks:
        return []

    query_embedding = (await embed_texts([query]))[0]
    scored = [
        {
            "chunk_id": c.id,
            "document_id": c.document_id,
            "content": c.content,
            "score": cosine_similarity(query_embedding, c.embedding or []),
        }
        for c in chunks
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def format_rag_context(results: list[dict]) -> str:
    if not results:
        return ""
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[Source {i} | doc:{r['document_id']} | score:{r['score']:.2f}]\n{r['content']}")
    return "\n\n# Retrieved context (cite as [Source N])\n" + "\n\n".join(blocks)
