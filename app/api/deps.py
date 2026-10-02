"""Dependency-injection singletons. Using @lru_cache means the (expensive)
embedding model and vector store connection are created once per process and
reused across requests, instead of per-request."""
from functools import lru_cache

from app.config import get_settings
from app.rag.agent import Agent, Tool
from app.rag.embeddings import get_embedder
from app.rag.llm import OllamaClient
from app.rag.pipeline import RAGPipeline
from app.rag.tools.calculator import calculator
from app.rag.tools.search_docs import make_search_docs_tool
from app.rag.vectorstore import VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    return VectorStore(persist_dir=settings.chroma_persist_dir, collection_name=settings.chroma_collection)


@lru_cache
def get_llm_client() -> OllamaClient:
    settings = get_settings()
    return OllamaClient(host=settings.ollama_host, model=settings.ollama_model, timeout=settings.ollama_timeout_seconds)


@lru_cache
def get_pipeline() -> RAGPipeline:
    settings = get_settings()
    return RAGPipeline(
        embedder=get_embedder(settings.embedding_model),
        vector_store=get_vector_store(),
        llm=get_llm_client(),
        top_k=settings.top_k,
    )


@lru_cache
def get_agent() -> Agent:
    settings = get_settings()
    pipeline = get_pipeline()
    tools = [
        Tool("search_docs", "Search the internal knowledge base for relevant passages. Input: a search query.",
             make_search_docs_tool(pipeline.retrieve)),
        Tool("calculator", "Evaluate a basic arithmetic expression. Input: a math expression like '12 * 4'.",
             calculator),
    ]
    return Agent(llm=get_llm_client(), tools=tools, max_steps=settings.agent_max_steps)
