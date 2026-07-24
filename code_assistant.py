"""
code_assistant.py
Prompt templates + execution for the code assistant feature set:
generate, explain, debug, refactor, review, and generate tests.
"""
from __future__ import annotations

from llm_providers import ChatMessage, get_provider

_SYSTEM = (
    "You are a senior software engineer acting as a code assistant. "
    "Always return clear, correct, production-quality code with brief explanations. "
    "Use fenced code blocks with the correct language tag."
)

_ACTION_PROMPTS = {
    "generate": "Write {language} code that does the following:\n{instructions}",
    "explain": "Explain what the following {language} code does, step by step:\n```{language}\n{code}\n```",
    "debug": (
        "Find and fix the bug(s) in the following {language} code. Explain the root cause, "
        "then give the corrected code:\n```{language}\n{code}\n```\nContext: {instructions}"
    ),
    "refactor": (
        "Refactor the following {language} code for readability, performance, and best practices. "
        "Explain the key changes:\n```{language}\n{code}\n```\nGoals: {instructions}"
    ),
    "review": (
        "Perform a thorough code review of the following {language} code. Cover correctness, "
        "security, performance, and style. Use a bullet list:\n```{language}\n{code}\n```"
    ),
    "test": (
        "Write comprehensive unit tests for the following {language} code using the standard "
        "testing framework for that language:\n```{language}\n{code}\n```"
    ),
}


async def run_code_action(
    action: str, code: str, instructions: str, language: str, provider_name: str
) -> str:
    template = _ACTION_PROMPTS[action]
    prompt = template.format(language=language, code=code, instructions=instructions or "N/A")
    provider = get_provider(provider_name)
    result = await provider.complete(
        [ChatMessage(role="system", content=_SYSTEM), ChatMessage(role="user", content=prompt)]
    )
    return result.text
