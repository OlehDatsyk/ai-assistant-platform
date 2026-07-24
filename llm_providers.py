"""
llm_providers.py
A unified async interface over OpenAI, Anthropic, and Google Gemini so the rest
of the application never has to know which vendor SDK is behind "the model".

Design: each concrete provider implements `LLMProvider` (stream + tool calling).
`get_provider(name)` is the single factory used everywhere else in the app.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ToolSpec:
    """Provider-agnostic tool/function definition."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema


@dataclass
class ChatMessage:
    role: str  # system | user | assistant | tool
    content: str
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionResult:
    text: str
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    raw: Any = None


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Yield text deltas as they arrive."""
        ...

    @abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.7,
    ) -> CompletionResult:
        """Non-streaming completion; used for tool-calling round trips and structured output."""
        ...

    def is_configured(self) -> bool:
        return True


# ---------------------------------------------------------------- OpenAI ----
class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=settings.openai_api_key or "missing")
        self._model = settings.openai_model

    def is_configured(self) -> bool:
        return bool(settings.openai_api_key)

    @staticmethod
    def _to_openai_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            entry: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.role == "tool":
                entry["tool_call_id"] = m.tool_call_id
                entry["content"] = m.content
            out.append(entry)
        return out

    @staticmethod
    def _to_openai_tools(tools: list[ToolSpec] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
            }
            for t in tools
        ]

    async def stream_chat(self, messages, tools=None, temperature=0.7) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=self._to_openai_messages(messages),
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    async def complete(self, messages, tools=None, temperature=0.7) -> CompletionResult:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=self._to_openai_messages(messages),
            tools=self._to_openai_tools(tools),
            temperature=temperature,
        )
        choice = resp.choices[0]
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    ToolCallRequest(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments or "{}"),
                    )
                )
        return CompletionResult(text=choice.message.content or "", tool_calls=tool_calls, raw=resp)


# --------------------------------------------------------------- Claude -----
class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self) -> None:
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key or "missing")
        self._model = settings.anthropic_model

    def is_configured(self) -> bool:
        return bool(settings.anthropic_api_key)

    @staticmethod
    def _split_system(messages: list[ChatMessage]) -> tuple[str, list[dict[str, Any]]]:
        system_parts = [m.content for m in messages if m.role == "system"]
        rest = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        return "\n".join(system_parts), rest

    @staticmethod
    def _to_anthropic_tools(tools: list[ToolSpec] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
        ]

    async def stream_chat(self, messages, tools=None, temperature=0.7) -> AsyncIterator[str]:
        system, chat_messages = self._split_system(messages)
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=4096,
            temperature=temperature,
            system=system or "You are a helpful assistant.",
            messages=chat_messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def complete(self, messages, tools=None, temperature=0.7) -> CompletionResult:
        system, chat_messages = self._split_system(messages)
        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            temperature=temperature,
            system=system or "You are a helpful assistant.",
            messages=chat_messages,
            tools=self._to_anthropic_tools(tools) or [],
        )
        text_parts = []
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCallRequest(id=block.id, name=block.name, arguments=block.input))
        return CompletionResult(text="".join(text_parts), tool_calls=tool_calls, raw=resp)


# --------------------------------------------------------------- Gemini -----
class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self) -> None:
        import google.generativeai as genai

        if settings.google_api_key:
            genai.configure(api_key=settings.google_api_key)
        self._genai = genai
        self._model_name = settings.gemini_model

    def is_configured(self) -> bool:
        return bool(settings.google_api_key)

    @staticmethod
    def _split_system(messages: list[ChatMessage]) -> tuple[str, list[dict[str, Any]]]:
        system_parts = [m.content for m in messages if m.role == "system"]
        history = []
        for m in messages:
            if m.role == "user":
                history.append({"role": "user", "parts": [m.content]})
            elif m.role == "assistant":
                history.append({"role": "model", "parts": [m.content]})
        return "\n".join(system_parts), history

    async def stream_chat(self, messages, tools=None, temperature=0.7) -> AsyncIterator[str]:
        system, history = self._split_system(messages)
        model = self._genai.GenerativeModel(self._model_name, system_instruction=system or None)
        convo = history[:-1]
        last = history[-1]["parts"][0] if history else ""
        chat = model.start_chat(history=convo)
        response = await chat.send_message_async(
            last, generation_config={"temperature": temperature}, stream=True
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text

    async def complete(self, messages, tools=None, temperature=0.7) -> CompletionResult:
        system, history = self._split_system(messages)
        model = self._genai.GenerativeModel(self._model_name, system_instruction=system or None)
        convo = history[:-1]
        last = history[-1]["parts"][0] if history else ""
        chat = model.start_chat(history=convo)
        response = await chat.send_message_async(last, generation_config={"temperature": temperature})
        return CompletionResult(text=response.text or "", tool_calls=[], raw=response)


_PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}
_instances: dict[str, LLMProvider] = {}


def get_provider(name: str) -> LLMProvider:
    """Factory + cache for provider instances. Raises ValueError on unknown name."""
    if name not in _PROVIDER_REGISTRY:
        raise ValueError(f"Unknown model provider: {name}")
    if name not in _instances:
        _instances[name] = _PROVIDER_REGISTRY[name]()
    return _instances[name]
