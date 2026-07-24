"""
tools.py
Central registry of callable "tools" exposed to the LLM via function/tool calling,
and reused as the tool implementations behind the MCP server.
"""
from __future__ import annotations

import ast
import operator
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from config import settings
from llm_providers import ToolSpec
from logging_config import get_logger

logger = get_logger(__name__)

ToolFunc = Callable[..., Awaitable[dict[str, Any]]]


# ------------------------------------------------------------- Web search ---
async def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web via Tavily if configured; otherwise return a helpful stub result."""
    if not settings.tavily_api_key:
        return {
            "results": [],
            "note": (
                "Web search is not configured. Set TAVILY_API_KEY in .env to enable "
                "live internet search."
            ),
        }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={"api_key": settings.tavily_api_key, "query": query, "max_results": max_results},
        )
        resp.raise_for_status()
        data = resp.json()
    results = [
        {"title": r.get("title"), "url": r.get("url"), "snippet": r.get("content")}
        for r in data.get("results", [])
    ]
    return {"results": results}


# -------------------------------------------------------------- Calculator --
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


async def calculator(expression: str) -> dict[str, Any]:
    """Safely evaluate a basic arithmetic expression (no eval() of arbitrary code)."""
    try:
        tree = ast.parse(expression, mode="eval")
        return {"result": _safe_eval(tree.body)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not evaluate expression: {exc}"}


# --------------------------------------------------------------- Registry ---
TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="web_search",
        description="Search the internet for current information and return a list of results.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="calculator",
        description="Evaluate a basic arithmetic expression, e.g. '2 * (3 + 4)'.",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    ),
]

TOOL_IMPLEMENTATIONS: dict[str, ToolFunc] = {
    "web_search": web_search,
    "calculator": calculator,
}


async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    func = TOOL_IMPLEMENTATIONS.get(name)
    if func is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return await func(**arguments)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Tool %s failed", name)
        return {"error": str(exc)}
