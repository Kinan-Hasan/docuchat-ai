"""Wraps the retriever as an agent tool, so the agent can explicitly decide
to search the knowledge base rather than it being forced on every turn."""
from __future__ import annotations

from typing import Callable


def make_search_docs_tool(retrieve_fn: Callable[[str], list[dict]]) -> Callable[[str], str]:
    def search_docs(query: str) -> str:
        results = retrieve_fn(query)
        if not results:
            return "No matching documents found."
        lines = [f"- ({r['metadata'].get('source', 'unknown')}) {r['content'][:200]}" for r in results]
        return "\n".join(lines)

    return search_docs
