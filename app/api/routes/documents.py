import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile

from app.api.deps import get_vector_store
from app.config import Settings, get_settings
from app.core.security import verify_api_key
from app.models.schemas import IngestResponse
from app.rag.embeddings import get_embedder
from app.rag.ingestion import load_and_chunk_documents
from app.rag.vectorstore import VectorStore

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/ingest", response_model=IngestResponse, dependencies=[Depends(verify_api_key)])
async def ingest_documents(
    files: list[UploadFile],
    vector_store: VectorStore = Depends(get_vector_store),
    settings: Settings = Depends(get_settings),
):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for f in files:
            dest = tmp_path / f.filename
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)

        chunks = load_and_chunk_documents(tmp_path, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        embedder = get_embedder(settings.embedding_model)
        embeddings = embedder.embed([c.content for c in chunks])
        vector_store.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
    return IngestResponse(documents_ingested=len(files), chunks_created=len(chunks))


@router.get("/count", dependencies=[Depends(verify_api_key)])
async def document_count(vector_store: VectorStore = Depends(get_vector_store)):
    return {"chunks_in_store": vector_store.count()}
