"""Document loading and chunking.

This is written by hand (no LangChain text splitter) on purpose: it's the
part of a RAG system interviewers most often probe on ("how did you decide
your chunk size / overlap / boundary strategy?"), so it's worth owning.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md"}


@dataclass
class Chunk:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)


def _split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries. Deliberately simple (regex, not a full
    NLP sentence tokenizer) — good enough for policy docs / wikis and keeps
    the dependency footprint small."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Recursive-ish character chunking that respects sentence boundaries
    where possible, with a sliding overlap so context isn't lost at chunk
    edges. Returns a list of chunk strings.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1
        if current_len + sentence_len > chunk_size and current:
            chunk_str = " ".join(current)
            chunks.append(chunk_str)

            # Build overlap: walk back from the end of the current chunk
            # until we've accumulated ~`overlap` characters worth of context.
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                # Always keep at least one sentence of overlap context, even
                # if that single sentence alone exceeds the nominal overlap
                # size — some context beats none at a chunk boundary.
                if overlap_sentences and overlap_len + len(s) > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s) + 1
            current = overlap_sentences
            current_len = overlap_len

        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(" ".join(current))

    # Fallback: if a single sentence is longer than chunk_size, hard-split it.
    final_chunks: list[str] = []
    for c in chunks:
        if len(c) <= chunk_size * 1.5:
            final_chunks.append(c)
        else:
            for i in range(0, len(c), chunk_size - overlap):
                final_chunks.append(c[i : i + chunk_size])

    return final_chunks


def _make_chunk_id(source: str, index: int, content: str) -> str:
    digest = hashlib.sha1(f"{source}:{index}:{content}".encode()).hexdigest()[:12]
    return f"{Path(source).stem}-{index}-{digest}"


def load_and_chunk_documents(
    directory: str | Path,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Load every supported document under `directory` and split it into
    Chunk objects ready for embedding."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    all_chunks: list[Chunk] = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        pieces = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(
                    id=_make_chunk_id(str(path), i, piece),
                    content=piece,
                    metadata={"source": path.name, "chunk_index": i},
                )
            )
    return all_chunks
