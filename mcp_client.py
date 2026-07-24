"""
mcp_client.py
Example MCP client that launches mcp_server.py as a subprocess over stdio and
demonstrates listing + calling tools, reading resources, and fetching prompts.

Run directly:
    python mcp_client.py
"""
from __future__ import annotations

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


async def demo() -> None:
    params = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools.tools])

            calc = await session.call_tool("calculator", {"expression": "(4 + 6) * 2"})
            print("calculator(4+6)*2 ->", calc.content[0].text)

            resources = await session.list_resources()
            print("Available resources:", [r.uri for r in resources.resources])

            about = await session.read_resource("platform://about")
            print("Resource platform://about ->", about.contents[0].text)

            prompts = await session.list_prompts()
            print("Available prompts:", [p.name for p in prompts.prompts])

            prompt = await session.get_prompt("summarize_document", {"text": "MCP standardizes tool use."})
            print("Prompt result ->", prompt.messages[0].content.text)


if __name__ == "__main__":
    asyncio.run(demo())
