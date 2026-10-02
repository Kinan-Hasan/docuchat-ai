import pytest
from pydantic import ValidationError

from app.models.schemas import ChatRequest, Source


def test_chat_request_valid():
    req = ChatRequest(query="What is the remote work policy?")
    assert req.top_k is None


def test_chat_request_rejects_empty_query():
    with pytest.raises(ValidationError):
        ChatRequest(query="")


def test_chat_request_rejects_too_long_query():
    with pytest.raises(ValidationError):
        ChatRequest(query="x" * 3000)


def test_chat_request_top_k_bounds():
    with pytest.raises(ValidationError):
        ChatRequest(query="hello", top_k=0)
    with pytest.raises(ValidationError):
        ChatRequest(query="hello", top_k=100)


def test_source_model():
    s = Source(content="text", source="doc.md", chunk_id="abc-1", score=0.87)
    assert s.score == 0.87
