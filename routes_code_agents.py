"""routes_code_agents.py - code assistant, AI agents, and model-listing endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends

import agents
import code_assistant
from auth import get_current_user
from config import settings
from models import User
from schemas import AgentRunRequest, CodeRequest

router = APIRouter(prefix="/api", tags=["code-agents"])


@router.post("/code/run")
async def run_code_action(payload: CodeRequest, user: User = Depends(get_current_user)):
    text = await code_assistant.run_code_action(
        action=payload.action,
        code=payload.code,
        instructions=payload.instructions,
        language=payload.language,
        provider_name=payload.model_provider,
    )
    return {"result": text}


@router.post("/agents/run")
async def run_agent(payload: AgentRunRequest, user: User = Depends(get_current_user)):
    result = await agents.run_agent(payload.agent, payload.goal, payload.model_provider)
    return {
        "final_answer": result.final_answer,
        "steps": [
            {"step": s.step, "thought": s.thought, "tool_calls": s.tool_calls} for s in result.steps
        ],
    }


@router.get("/models")
async def list_models(user: User = Depends(get_current_user)):
    return {
        "providers": [
            {"id": "openai", "label": "OpenAI GPT", "configured": settings.providers_available["openai"]},
            {"id": "anthropic", "label": "Anthropic Claude", "configured": settings.providers_available["anthropic"]},
            {"id": "gemini", "label": "Google Gemini", "configured": settings.providers_available["gemini"]},
        ]
    }
