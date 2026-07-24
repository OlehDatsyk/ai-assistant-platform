"""
agents.py
A small but real multi-step agent loop: plan -> act (tool calls) -> observe -> repeat
until the model signals it is done or a step limit is hit. Three example agents
(research, coding, email) are thin persona wrappers around the same loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from llm_providers import ChatMessage, get_provider
from logging_config import get_logger
from tools import TOOL_SPECS, call_tool

logger = get_logger(__name__)

MAX_STEPS = 6

_PERSONAS = {
    "research": (
        "You are a research agent. Break the goal into sub-questions, use the web_search "
        "tool to gather evidence, and synthesize a well-cited answer. Think step by step."
    ),
    "coding": (
        "You are a coding agent. Plan the implementation, reason about edge cases, and "
        "produce working code with a short explanation. Use tools if they help verify logic."
    ),
    "email": (
        "You are an email assistant agent. Draft a clear, professional email that achieves "
        "the stated goal. If information is missing, state your assumptions explicitly."
    ),
}


@dataclass
class AgentStep:
    step: int
    thought: str
    tool_calls: list[dict] = field(default_factory=list)


@dataclass
class AgentRunResult:
    final_answer: str
    steps: list[AgentStep] = field(default_factory=list)


async def run_agent(agent: str, goal: str, provider_name: str = "openai") -> AgentRunResult:
    persona = _PERSONAS.get(agent, _PERSONAS["research"])
    provider = get_provider(provider_name)

    messages = [
        ChatMessage(role="system", content=persona + " When you have the final answer, prefix it with 'FINAL:'."),
        ChatMessage(role="user", content=goal),
    ]

    steps: list[AgentStep] = []
    for i in range(MAX_STEPS):
        result = await provider.complete(messages, tools=TOOL_SPECS)

        if not result.tool_calls:
            text = result.text.strip()
            final = text.split("FINAL:", 1)[-1].strip() if "FINAL:" in text else text
            steps.append(AgentStep(step=i, thought=text))
            return AgentRunResult(final_answer=final, steps=steps)

        messages.append(ChatMessage(role="assistant", content=result.text or ""))
        tool_log = []
        for call in result.tool_calls:
            tool_result = await call_tool(call.name, call.arguments)
            tool_log.append({"tool": call.name, "arguments": call.arguments, "result": tool_result})
            messages.append(
                ChatMessage(role="tool", content=str(tool_result), tool_call_id=call.id, name=call.name)
            )
        steps.append(AgentStep(step=i, thought=result.text or "(tool call)", tool_calls=tool_log))

    return AgentRunResult(
        final_answer="Agent reached the maximum number of steps without a final answer.",
        steps=steps,
    )
