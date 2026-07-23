from pathlib import Path

import pytest

from semantic_notes.ingestion.document_loader import (
    MarkdownDocumentLoader,
)


def test_load_markdown_document(tmp_path: Path) -> None:
    note_path = tmp_path / "spark.md"

    note_path.write_text(
        "# Spark Watermarks\n\nWatermarks handle late data.",
        encoding="utf-8",
    )

    loader = MarkdownDocumentLoader()
    document = loader.load_file(note_path)

    assert document.title == "Spark Watermarks"
    assert document.source_path == note_path
    assert "late data" in document.content


def test_use_filename_when_heading_is_missing(
    tmp_path: Path,
) -> None:
    note_path = tmp_path / "airflow-retries.md"

    note_path.write_text(
        "Airflow supports task retries.",
        encoding="utf-8",
    )

    loader = MarkdownDocumentLoader()
    document = loader.load_file(note_path)

    assert document.title == "Airflow Retries"


def test_reject_empty_document(tmp_path: Path) -> None:
    note_path = tmp_path / "empty.md"
    note_path.write_text("", encoding="utf-8")

    loader = MarkdownDocumentLoader()

    with pytest.raises(ValueError, match="Document is empty"):
        loader.load_file(note_path)
