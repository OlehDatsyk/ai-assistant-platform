"""routes_pages.py — server-rendered dashboard pages (HTML shell; data loads via JS/API)."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from config import BASE_DIR

router = APIRouter(tags=["pages"])
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

PAGES = {
    "/": "home.html",
    "/chat": "chat.html",
    "/documents": "documents.html",
    "/memory": "memory.html",
    "/calendar": "calendar.html",
    "/emails": "emails.html",
    "/telegram": "telegram.html",
    "/voice": "voice.html",
    "/vision": "vision.html",
    "/settings": "settings.html",
    "/prompts": "prompts.html",
    "/agents": "agents.html",
    "/code": "code.html",
    "/login": "login.html",
}


def _register(path: str, template: str) -> None:
    @router.get(path, include_in_schema=False)
    async def _page(request: Request, template=template):  # noqa: ANN001
        return templates.TemplateResponse(template, {"request": request, "active_path": path})

    _page.__name__ = f"page_{template}"


for _path, _template in PAGES.items():
    _register(_path, _template)
