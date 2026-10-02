"""Embedding model wrapper. Uses sentence-transformers locally (no API cost,
runs on CPU) — swap `model_name` for a bigger model if you need better recall
and have the compute for it."""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


@lru_cache
def get_embedder(model_name: str) -> Embedder:
    """Cached so the (relatively expensive) model load happens once per process."""
    return Embedder(model_name)
