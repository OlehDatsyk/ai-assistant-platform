"""
schemas.py
Pydantic v2 models used for API request/response validation across the app.
"""
from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

# `model_provider` fields are intentional and harmless; silence pydantic's
# protected-namespace warning for this module rather than renaming a clear API field.
warnings.filterwarnings(
    "ignore", message='Field "model_provider".*protected namespace.*', category=UserWarning
)

ModelProvider = Literal["openai", "anthropic", "gemini"]


# ---------- Auth ----------
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str = ""


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    email: str
    full_name: str
    created_at: datetime


# ---------- Conversations ----------
class ConversationCreate(BaseModel):
    title: str = "New Conversation"
    model_provider: ModelProvider = "openai"
    system_prompt: str = "You are a helpful AI assistant."


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    title: str
    model_provider: str
    system_prompt: str
    is_pinned: bool
    created_at: datetime
    updated_at: datetime


class MessageCreate(BaseModel):
    content: str
    model_provider: ModelProvider | None = None
    use_rag: bool = False
    document_ids: list[str] = Field(default_factory=list)
    use_web_search: bool = False


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    role: str
    content: str
    citations: dict | None = None
    created_at: datetime


# ---------- Memory ----------
class MemoryCreate(BaseModel):
    content: str
    category: str = "general"
    importance: int = Field(default=1, ge=1, le=5)


class MemoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    content: str
    category: str
    importance: int
    created_at: datetime


# ---------- Prompt library ----------
class PromptCreate(BaseModel):
    title: str
    content: str
    category: str = "general"


class PromptOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    title: str
    content: str
    category: str
    is_favorite: bool
    created_at: datetime


# ---------- Documents / RAG ----------
class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    filename: str
    file_type: str
    size_bytes: int
    status: str
    created_at: datetime


class RagQuery(BaseModel):
    query: str
    document_ids: list[str] = Field(default_factory=list)
    top_k: int = 5


# ---------- Code assistant ----------
class CodeRequest(BaseModel):
    action: Literal["generate", "explain", "debug", "refactor", "review", "test"]
    code: str = ""
    instructions: str = ""
    language: str = "python"
    model_provider: ModelProvider = "openai"


# ---------- Calendar ----------
class EventCreate(BaseModel):
    summary: str
    description: str = ""
    start_iso: str
    end_iso: str
    attendees: list[str] = Field(default_factory=list)


# ---------- Agents ----------
class AgentRunRequest(BaseModel):
    agent: Literal["research", "coding", "email"]
    goal: str
    model_provider: ModelProvider = "openai"


# ---------- Generic tool/structured output ----------
class StructuredExtractRequest(BaseModel):
    text: str
    schema_description: str
    model_provider: ModelProvider = "openai"


class ChatStreamChunk(BaseModel):
    delta: str
    done: bool = False
    citations: list[dict[str, Any]] | None = None
