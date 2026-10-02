from fastapi import APIRouter, Depends

from app.api.deps import get_llm_client, get_vector_store
from app.config import Settings, get_settings
from app.models.schemas import HealthResponse
from app.rag.llm import OllamaClient
from app.rag.vectorstore import VectorStore

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    vector_store: VectorStore = Depends(get_vector_store),
    llm: OllamaClient = Depends(get_llm_client),
    settings: Settings = Depends(get_settings),
):
    ollama_ok = await llm.is_reachable()
    return HealthResponse(
        status="ok",
        vector_store_documents=vector_store.count(),
        ollama_reachable=ollama_ok,
        ollama_model=settings.ollama_model,
    )
