"""Pydantic request/response models. These double as the API's contract and
its input validation layer."""
from pydantic import BaseModel, Field


class Source(BaseModel):
    content: str
    source: str
    chunk_id: str
    score: float = Field(..., description="Similarity score, higher = more relevant")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: float
    model: str


class AgentStep(BaseModel):
    thought: str
    action: str
    action_input: str
    observation: str


class AgentChatResponse(BaseModel):
    answer: str
    steps: list[AgentStep]
    latency_ms: float


class IngestResponse(BaseModel):
    documents_ingested: int
    chunks_created: int


class HealthResponse(BaseModel):
    status: str
    vector_store_documents: int
    ollama_reachable: bool
    ollama_model: str
