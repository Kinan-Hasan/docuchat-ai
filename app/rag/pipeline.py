"""The RAG orchestration layer: retrieve -> build a grounded prompt -> generate.

This is intentionally framework-free so every step is visible and testable,
which is exactly what you want to be able to explain in an interview.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from app.rag.embeddings import Embedder
from app.rag.llm import OllamaClient
from app.rag.vectorstore import VectorStore

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "context provided below. If the answer isn't in the context, say you "
    "don't know rather than guessing. Cite sources inline using [1], [2] "
    "style markers that match the numbered context blocks."
)


class RAGPipeline:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, llm: OllamaClient, top_k: int = 4):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.default_top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        query_embedding = self.embedder.embed_one(query)
        return self.vector_store.query(query_embedding, top_k=top_k or self.default_top_k)

    @staticmethod
    def build_prompt(query: str, sources: list[dict]) -> str:
        context_blocks = []
        for i, s in enumerate(sources, start=1):
            context_blocks.append(f"[{i}] (source: {s['metadata'].get('source', 'unknown')})\n{s['content']}")
        context = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."
        return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    async def answer(self, query: str, top_k: int | None = None) -> dict:
        sources = self.retrieve(query, top_k=top_k)
        prompt = self.build_prompt(query, sources)
        response = await self.llm.generate(prompt, system=SYSTEM_PROMPT)
        return {"answer": response.strip(), "sources": sources}

    async def answer_stream(self, query: str, top_k: int | None = None) -> AsyncIterator[str]:
        sources = self.retrieve(query, top_k=top_k)
        prompt = self.build_prompt(query, sources)
        async for token in self.llm.generate_stream(prompt, system=SYSTEM_PROMPT):
            yield token
