from components.chunker_component import SemanticChunker


class _FakeEmbeddings:
    def embed_documents(self, texts):
        return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]


def test_recursive_chunking_produces_chunks(monkeypatch):
    monkeypatch.setattr("components.chunker_component.config.chunking.strategy", "recursive")
    monkeypatch.setattr("components.chunker_component.config.chunking.max_chunk_chars", 60)
    monkeypatch.setattr("components.chunker_component.config.chunking.min_chunk_chars", 10)
    monkeypatch.setattr("components.chunker_component.config.chunking.overlap_chars", 0)
    chunker = SemanticChunker(_FakeEmbeddings(), lambda *_args, **_kwargs: "[]")
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."

    chunks = chunker.chunk_text(text)

    assert chunks
    assert all(chunk.text for chunk in chunks)