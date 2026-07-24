"""test_rag_and_tools.py — unit tests for chunking, local embeddings, cosine search, and tools."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_service
from tools import call_tool


def test_chunk_text_respects_overlap():
    text = " ".join(str(i) for i in range(500))
    chunks = rag_service.chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 1
    # Every chunk except the last should be exactly `size` characters.
    for c in chunks[:-1]:
        assert len(c) == 100


def test_chunk_text_handles_empty_string():
    assert rag_service.chunk_text("") == []


def test_hash_embedding_is_deterministic_and_normalized():
    import numpy as np

    v1 = rag_service._hash_embedding("hello world")
    v2 = rag_service._hash_embedding("hello world")
    assert v1 == v2
    norm = np.linalg.norm(v1)
    assert abs(norm - 1.0) < 1e-4 or norm == 0


def test_cosine_similarity_identical_vectors_is_one():
    v = rag_service._hash_embedding("machine learning is fun")
    assert abs(rag_service.cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_different_vectors_lower():
    v1 = rag_service._hash_embedding("cats and dogs")
    v2 = rag_service._hash_embedding("quantum physics research")
    assert rag_service.cosine_similarity(v1, v2) < 0.99


@pytest.mark.asyncio
async def test_calculator_tool_basic():
    result = await call_tool("calculator", {"expression": "2 * (3 + 4)"})
    assert result == {"result": 14}


@pytest.mark.asyncio
async def test_calculator_tool_rejects_unsafe_expression():
    result = await call_tool("calculator", {"expression": "__import__('os').system('echo hi')"})
    assert "error" in result


@pytest.mark.asyncio
async def test_unknown_tool_returns_error():
    result = await call_tool("not_a_real_tool", {})
    assert "error" in result
