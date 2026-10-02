from app.rag.ingestion import _split_sentences, chunk_text, load_and_chunk_documents


def test_split_sentences_basic():
    text = "This is one sentence. This is another! Is this a third?"
    sentences = _split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "This is one sentence."


def test_split_sentences_empty():
    assert _split_sentences("") == []
    assert _split_sentences("   ") == []


def test_chunk_text_respects_chunk_size():
    text = " ".join([f"Sentence number {i}." for i in range(50)])
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 150  # allows a little slack for sentence boundaries


def test_chunk_text_creates_overlap():
    text = "Alpha bravo charlie. Delta echo foxtrot. Golf hotel india. Juliet kilo lima."
    chunks = chunk_text(text, chunk_size=40, overlap=15)
    assert len(chunks) >= 2
    # Some content from the end of chunk 1 should reappear at the start of chunk 2
    assert any(word in chunks[1] for word in chunks[0].split()[-3:])


def test_chunk_text_rejects_bad_overlap():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=50, overlap=50)


def test_load_and_chunk_documents(tmp_path):
    doc = tmp_path / "policy.md"
    doc.write_text("This is a test policy. It has two sentences.")
    chunks = load_and_chunk_documents(tmp_path, chunk_size=200, overlap=20)
    assert len(chunks) == 1
    assert chunks[0].metadata["source"] == "policy.md"
    assert "test policy" in chunks[0].content


def test_load_and_chunk_documents_missing_dir():
    import pytest
    with pytest.raises(FileNotFoundError):
        load_and_chunk_documents("/nonexistent/path/xyz")
