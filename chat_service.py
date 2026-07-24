"""
chat_service.py
Orchestrates a single chat turn: builds context (short/long-term memory, optional
RAG, optional web search), runs the tool-calling loop if needed, and streams the
final answer back to the caller.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

import memory_service
import rag_service
from llm_providers import ChatMessage, ToolCallRequest, get_provider
from logging_config import get_logger
from models import Conversation, Message
from tools import TOOL_SPECS, call_tool

logger = get_logger(__name__)

MAX_TOOL_ROUNDS = 3


async def _build_context_messages(
    db: AsyncSession,
    conversation: Conversation,
    rag_context: str = "",
) -> list[ChatMessage]:
    long_term = await memory_service.get_long_term_memory(db, conversation.user_id)
    memory_block = memory_service.format_memory_block(long_term)

    system_content = conversation.system_prompt + memory_block + rag_context
    history = await memory_service.get_short_term_context(db, conversation.id)

    messages = [ChatMessage(role="system", content=system_content)]
    for m in history:
        if m.role in ("user", "assistant"):
            messages.append(ChatMessage(role=m.role, content=m.content))
    return messages


async def run_tool_loop(
    provider_name: str, messages: list[ChatMessage], enable_tools: bool
) -> tuple[list[ChatMessage], list[dict]]:
    """Runs up to MAX_TOOL_ROUNDS of tool-calling before final generation. Returns
    updated messages plus a log of tool invocations (for optional citation display)."""
    if not enable_tools:
        return messages, []

    provider = get_provider(provider_name)
    tool_log: list[dict] = []

    for _ in range(MAX_TOOL_ROUNDS):
        result = await provider.complete(messages, tools=TOOL_SPECS)
        if not result.tool_calls:
            break
        # Record the assistant's tool-call turn, then execute each tool and append results.
        messages.append(ChatMessage(role="assistant", content=result.text or ""))
        for call in result.tool_calls:
            tool_result = await call_tool(call.name, call.arguments)
            tool_log.append({"tool": call.name, "arguments": call.arguments, "result": tool_result})
            messages.append(
                ChatMessage(
                    role="tool",
                    content=str(tool_result),
                    tool_call_id=call.id,
                    name=call.name,
                )
            )
    return messages, tool_log


async def generate_reply_stream(
    db: AsyncSession,
    conversation: Conversation,
    user_text: str,
    provider_name: str,
    use_rag: bool = False,
    document_ids: list[str] | None = None,
    use_web_search: bool = False,
) -> AsyncIterator[str]:
    """High-level streaming generator used by the /chat/stream endpoint."""
    rag_context = ""
    if use_rag:
        results = await rag_service.semantic_search(db, user_text, document_ids)
        rag_context = rag_service.format_rag_context(results)

    context_messages = await _build_context_messages(db, conversation, rag_context)
    context_messages.append(ChatMessage(role="user", content=user_text))

    if use_web_search:
        context_messages, _ = await run_tool_loop(provider_name, context_messages, enable_tools=True)

    provider = get_provider(provider_name)
    full_text = []
    async for delta in provider.stream_chat(context_messages):
        full_text.append(delta)
        yield delta

    # Persist both sides of the turn.
    db.add(Message(conversation_id=conversation.id, role="user", content=user_text))
    db.add(
        Message(conversation_id=conversation.id, role="assistant", content="".join(full_text))
    )
    await db.commit()

    # Fire-and-forget long-term memory extraction (best-effort, non-blocking semantics kept simple here).
    await memory_service.maybe_extract_memory(db, conversation.user_id, user_text, provider_name)
