from pathlib import Path

import pytest

from semantic_notes.ingestion.chunker import TextChunker
from semantic_notes.models import Document


def create_document(content: str) -> Document:
    return Document(
        source_path=Path("data/notes/test.md"),
        title="Test Note",
        content=content,
    )


def test_small_document_creates_one_chunk() -> None:
    chunker = TextChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = chunker.split_document(create_document("A short semantic search note."))

    assert len(chunks) == 1
    assert chunks[0].content == "A short semantic search note."
    assert chunks[0].chunk_index == 0


def test_large_document_creates_multiple_chunks() -> None:
    chunker = TextChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    content = "A" * 250

    chunks = chunker.split_document(create_document(content))

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 100 for chunk in chunks)


def test_chunk_ids_are_deterministic() -> None:
    chunker = TextChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    document = create_document("The same content should produce the same identifier.")

    first_result = chunker.split_document(document)
    second_result = chunker.split_document(document)

    assert first_result[0].chunk_id == second_result[0].chunk_id


def test_overlap_cannot_equal_chunk_size() -> None:
    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller",
    ):
        TextChunker(
            chunk_size=100,
            chunk_overlap=100,
        )
