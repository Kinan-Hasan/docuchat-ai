"""Thin wrapper around a persistent ChromaDB collection. Keeping this as its
own module means the rest of the app doesn't know or care which vector DB is
behind it — swapping to Qdrant/pgvector later only touches this file."""
from __future__ import annotations

import chromadb


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]) -> None:
        if not ids:
            return
        self._collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, query_embedding: list[float], top_k: int = 4) -> list[dict]:
        result = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)
        hits = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for id_, doc, meta, dist in zip(ids, docs, metas, distances):
            # Chroma returns cosine *distance*; convert to a similarity score in [0, 1].
            score = max(0.0, 1.0 - dist / 2.0)
            hits.append({"id": id_, "content": doc, "metadata": meta, "score": score})
        return hits

    def count(self) -> int:
        return self._collection.count()
