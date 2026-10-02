"""HTTP client for a local Ollama server. Talking to Ollama's plain REST API
(rather than a heavy framework's LLM abstraction) keeps this explicit and
easy to reason about, and it's trivial to point at a hosted model later by
swapping this one class."""
from __future__ import annotations

from collections.abc import AsyncIterator

import httpx


class OllamaClient:
    def __init__(self, host: str, model: str, timeout: int = 120):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def generate(self, prompt: str, system: str | None = None) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.host}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "")

    async def generate_stream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        payload = {"model": self.model, "prompt": prompt, "stream": True}
        if system:
            payload["system"] = system
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", f"{self.host}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    import json

                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break

    async def is_reachable(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                resp = await client.get(f"{self.host}/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
