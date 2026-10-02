"""Integration tests using FastAPI's TestClient. The RAG pipeline and agent
are mocked out via dependency overrides, so these tests run fast and don't
require Ollama or a real embedding model to be running."""
import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_llm_client, get_pipeline, get_vector_store
from app.main import app


class FakeVectorStore:
    def count(self):
        return 42


class FakeLLM:
    async def is_reachable(self):
        return True


class FakePipeline:
    async def answer(self, query, top_k=None):
        return {
            "answer": "You can work remotely three days per week. [1]",
            "sources": [
                {"id": "remote_work-0-abc123", "content": "...three days per week...",
                 "metadata": {"source": "remote_work_policy.md"}, "score": 0.91}
            ],
        }


@pytest.fixture
def client():
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    app.dependency_overrides[get_llm_client] = lambda: FakeLLM()
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["vector_store_documents"] == 42
    assert body["ollama_reachable"] is True


def test_chat_requires_api_key(client):
    resp = client.post("/chat", json={"query": "test"})
    assert resp.status_code == 401


def test_chat_with_valid_key(client):
    resp = client.post("/chat", json={"query": "How many remote days?"}, headers={"X-API-Key": "devkey123"})
    assert resp.status_code == 200
    body = resp.json()
    assert "three days" in body["answer"]
    assert len(body["sources"]) == 1
    assert body["sources"][0]["source"] == "remote_work_policy.md"


def test_chat_rejects_empty_query(client):
    resp = client.post("/chat", json={"query": ""}, headers={"X-API-Key": "devkey123"})
    assert resp.status_code == 422


def test_root_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "docs" in resp.json()
