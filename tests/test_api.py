"""test_api.py — integration tests hitting the FastAPI app through httpx's ASGI transport."""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_register_and_login(client):
    resp = await client.post(
        "/api/auth/register",
        json={"email": "alice@example.com", "password": "password123", "full_name": "Alice"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["email"] == "alice@example.com"

    resp = await client.post(
        "/api/auth/login", json={"email": "alice@example.com", "password": "password123"}
    )
    assert resp.status_code == 200
    assert "access_token" in resp.json()


@pytest.mark.asyncio
async def test_login_rejects_wrong_password(client):
    await client.post(
        "/api/auth/register",
        json={"email": "bob@example.com", "password": "password123", "full_name": "Bob"},
    )
    resp = await client.post(
        "/api/auth/login", json={"email": "bob@example.com", "password": "wrong-password"}
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_protected_route_requires_auth(client):
    resp = await client.get("/api/auth/me")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_conversation_crud(client, auth_headers):
    resp = await client.post(
        "/api/chat/conversations",
        json={"title": "My chat", "model_provider": "openai"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    conv = resp.json()
    assert conv["title"] == "My chat"

    resp = await client.get("/api/chat/conversations", headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 1

    resp = await client.patch(f"/api/chat/conversations/{conv['id']}/pin", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["is_pinned"] is True

    resp = await client.delete(f"/api/chat/conversations/{conv['id']}", headers=auth_headers)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_memory_crud(client, auth_headers):
    resp = await client.post(
        "/api/memory",
        json={"content": "Prefers dark mode", "category": "preference", "importance": 3},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    item = resp.json()

    resp = await client.get("/api/memory", headers=auth_headers)
    assert len(resp.json()) == 1

    resp = await client.delete(f"/api/memory/{item['id']}", headers=auth_headers)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_prompt_library_crud(client, auth_headers):
    resp = await client.post(
        "/api/prompts",
        json={"title": "Summarizer", "content": "Summarize this: {text}", "category": "general"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    prompt = resp.json()
    assert prompt["is_favorite"] is False

    resp = await client.patch(f"/api/prompts/{prompt['id']}/favorite", headers=auth_headers)
    assert resp.json()["is_favorite"] is True


@pytest.mark.asyncio
async def test_settings_defaults_and_update(client, auth_headers):
    resp = await client.get("/api/settings", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["default_model"] == "openai"

    resp = await client.put("/api/settings", json={"theme": "light"}, headers=auth_headers)
    assert resp.json()["theme"] == "light"


@pytest.mark.asyncio
async def test_models_endpoint_lists_all_three_providers(client, auth_headers):
    resp = await client.get("/api/models", headers=auth_headers)
    ids = {p["id"] for p in resp.json()["providers"]}
    assert ids == {"openai", "anthropic", "gemini"}


@pytest.mark.asyncio
async def test_health_check(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
