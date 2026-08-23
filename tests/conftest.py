"""conftest.py - shared pytest fixtures: isolated test DB + FastAPI test client."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from httpx import ASGITransport, AsyncClient  # noqa: E402

import database  # noqa: E402
from database import Base, engine  # noqa: E402


@pytest_asyncio.fixture
async def app():
    from main import app as fastapi_app

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield fastapi_app

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def auth_headers(client):
    await client.post(
        "/api/auth/register",
        json={"email": "pytest@example.com", "password": "password123", "full_name": "Pytest User"},
    )
    resp = await client.post(
        "/api/auth/login", json={"email": "pytest@example.com", "password": "password123"}
    )
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
