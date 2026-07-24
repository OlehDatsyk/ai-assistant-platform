"""
mcp_server.py
A standalone MCP (Model Context Protocol) server demonstrating the three MCP
primitives: Tools, Resources, and Prompts. Run directly:

    python mcp_server.py

It reuses the same tool implementations as the in-app chat (tools.py) so the
same logic is exposed both to this app's own LLM calls AND to any external
MCP-compatible client (Claude Desktop, another agent, etc).
"""
from __future__ import annotations

import asyncio
import logging
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import GetPromptResult, Prompt, PromptArgument, PromptMessage, Resource, TextContent, Tool

from tools import TOOL_SPECS, call_tool

# IMPORTANT: the stdio MCP transport reserves stdout exclusively for JSON-RPC
# protocol messages. The shared logging_config module logs to stdout, which
# would corrupt the protocol stream, so this server configures its own
# stderr-only logger instead of calling configure_logging().
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = Server("ai-assistant-platform-mcp")


# ------------------------------------------------------------------ Tools ---
@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(name=t.name, description=t.description, inputSchema=t.parameters) for t in TOOL_SPECS
    ]


@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    result = await call_tool(name, arguments or {})
    return [TextContent(type="text", text=str(result))]


# -------------------------------------------------------------- Resources ---
_RESOURCES = {
    "platform://about": "AI Assistant Platform: a multi-model, multi-agent assistant with RAG, "
    "memory, MCP, and tool calling built on FastAPI.",
    "platform://changelog": "See CHANGELOG.md in the repository root for release history.",
}


@app.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(uri=uri, name=uri.split("//")[-1], description="Static platform resource", mimeType="text/plain")
        for uri in _RESOURCES
    ]


@app.read_resource()
async def read_resource(uri) -> str:
    # The SDK passes a pydantic AnyUrl, not a plain str, so compare by string form.
    return _RESOURCES.get(str(uri), "Resource not found.")


# ---------------------------------------------------------------- Prompts ---
@app.list_prompts()
async def list_prompts() -> list[Prompt]:
    return [
        Prompt(
            name="summarize_document",
            description="Summarize a piece of text into key bullet points",
            arguments=[PromptArgument(name="text", description="Text to summarize", required=True)],
        ),
        Prompt(
            name="code_review",
            description="Perform a structured code review",
            arguments=[PromptArgument(name="code", description="Code to review", required=True)],
        ),
    ]


@app.get_prompt()
async def get_prompt(name: str, arguments: dict) -> GetPromptResult:
    if name == "summarize_document":
        text = arguments.get("text", "")
        content = f"Summarize the following text into 5 concise bullet points:\n\n{text}"
    elif name == "code_review":
        code = arguments.get("code", "")
        content = f"Review this code for correctness, security, and style:\n\n{code}"
    else:
        content = "Unknown prompt."
    return GetPromptResult(
        description=f"Generated prompt for '{name}'",
        messages=[PromptMessage(role="user", content=TextContent(type="text", text=content))],
    )


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
