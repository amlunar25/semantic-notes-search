from pathlib import Path

from semantic_notes.ingestion.fingerprint import (
    calculate_content_hash,
)
from semantic_notes.models import Document


def create_document(content: str) -> Document:
    return Document(
        source_path=Path("data/notes/test.md"),
        title="Test",
        content=content,
    )


def test_same_content_produces_same_hash() -> None:
    first = calculate_content_hash(create_document("Spark handles data."))

    second = calculate_content_hash(create_document("Spark handles data."))

    assert first == second


def test_different_content_produces_different_hash() -> None:
    first = calculate_content_hash(create_document("Spark handles data."))

    second = calculate_content_hash(create_document("Spark handles streaming data."))

    assert first != second


def test_line_endings_are_normalized() -> None:
    unix_hash = calculate_content_hash(create_document("Line one\nLine two"))

    windows_hash = calculate_content_hash(create_document("Line one\r\nLine two"))

    assert unix_hash == windows_hash


def test_trailing_spaces_are_ignored() -> None:
    clean_hash = calculate_content_hash(create_document("Line one\nLine two"))

    spaced_hash = calculate_content_hash(create_document("Line one   \nLine two   "))

    assert clean_hash == spaced_hash
