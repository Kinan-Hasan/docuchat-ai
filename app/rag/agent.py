"""A small ReAct-style agent: the model reasons in a loop, choosing between
tools and a final answer, expressed as JSON. This is implemented by hand
(prompt-based tool calling) rather than relying on a provider's native
function-calling API, because most local models via Ollama don't support
that reliably — this pattern works with any instruction-following model.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from app.rag.llm import OllamaClient

AGENT_SYSTEM_PROMPT = """You are an assistant that can use tools to answer questions.
Available tools:
{tool_descriptions}

Respond with ONLY a single JSON object, no other text, in one of these forms:
{{"thought": "...", "action": "<tool_name>", "action_input": "..."}}
{{"thought": "...", "action": "final_answer", "action_input": "..."}}

Always think step by step in "thought" before choosing an action.
Use final_answer as soon as you have enough information."""


class Tool:
    def __init__(self, name: str, description: str, func: Callable[[str], str]):
        self.name = name
        self.description = description
        self.func = func


class AgentStepResult:
    def __init__(self, thought: str, action: str, action_input: str, observation: str):
        self.thought = thought
        self.action = action
        self.action_input = action_input
        self.observation = observation


def _extract_json(text: str) -> dict:
    """Models sometimes wrap JSON in prose or code fences — pull out the
    first {...} block and parse it defensively."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    return json.loads(match.group(0))


class Agent:
    def __init__(self, llm: OllamaClient, tools: list[Tool], max_steps: int = 5):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

    def _system_prompt(self) -> str:
        descriptions = "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())
        return AGENT_SYSTEM_PROMPT.format(tool_descriptions=descriptions)

    async def run(self, query: str) -> tuple[str, list[AgentStepResult]]:
        transcript = f"Question: {query}"
        steps: list[AgentStepResult] = []

        for _ in range(self.max_steps):
            raw = await self.llm.generate(transcript, system=self._system_prompt())
            try:
                parsed = _extract_json(raw)
            except ValueError:
                return raw.strip(), steps

            thought = parsed.get("thought", "")
            action = parsed.get("action", "final_answer")
            action_input = str(parsed.get("action_input", ""))

            if action == "final_answer":
                steps.append(AgentStepResult(thought, action, action_input, "-"))
                return action_input, steps

            tool = self.tools.get(action)
            observation = f"error: unknown tool '{action}'" if tool is None else tool.func(action_input)
            steps.append(AgentStepResult(thought, action, action_input, observation))
            transcript += f"\nThought: {thought}\nAction: {action}({action_input})\nObservation: {observation}"

        return "I could not reach a final answer within the step limit.", steps
