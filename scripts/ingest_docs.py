"""CLI to ingest a directory of documents into the vector store.

Usage:
    python -m scripts.ingest_docs --path data/sample_docs
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.rag.embeddings import get_embedder
from app.rag.ingestion import load_and_chunk_documents
from app.rag.vectorstore import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the DocuChat vector store")
    parser.add_argument("--path", required=True, help="Directory of .txt/.md files to ingest")
    args = parser.parse_args()

    settings = get_settings()
    print(f"Loading and chunking documents from {args.path} ...")
    chunks = load_and_chunk_documents(args.path, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    print(f"  -> {len(chunks)} chunks created")

    print(f"Embedding chunks with '{settings.embedding_model}' ...")
    start = time.time()
    embedder = get_embedder(settings.embedding_model)
    embeddings = embedder.embed([c.content for c in chunks])
    print(f"  -> embedded in {time.time() - start:.1f}s")

    print(f"Writing to vector store at '{settings.chroma_persist_dir}' ...")
    store = VectorStore(persist_dir=settings.chroma_persist_dir, collection_name=settings.chroma_collection)
    store.add(
        ids=[c.id for c in chunks],
        embeddings=embeddings,
        documents=[c.content for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )
    print(f"Done. Vector store now has {store.count()} chunks.")


if __name__ == "__main__":
    main()
