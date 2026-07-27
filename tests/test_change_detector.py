from pathlib import Path

from semantic_notes.ingestion.change_detector import (
    DocumentChangeDetector,
)
from semantic_notes.ingestion.fingerprint import (
    calculate_content_hash,
)
from semantic_notes.ingestion.paths import normalize_source_path
from semantic_notes.models import (
    Document,
    DocumentStatus,
    ManifestEntry,
)


def create_document(
    filename: str,
    content: str,
) -> Document:
    return Document(
        source_path=Path(filename),
        title="Test",
        content=content,
    )


def create_manifest_entry(
    document: Document,
) -> ManifestEntry:
    source = normalize_source_path(document.source_path)

    return ManifestEntry(
        source=source,
        document_id="document-123",
        content_hash=calculate_content_hash(document),
        chunk_count=1,
    )


def test_new_document_is_detected() -> None:
    document = create_document(
        "data/notes/new.md",
        "New document",
    )

    detector = DocumentChangeDetector()

    changes = detector.detect(
        documents=[document],
        previous_manifest={},
    )

    assert len(changes) == 1
    assert changes[0].status == DocumentStatus.NEW


def test_unchanged_document_is_detected() -> None:
    document = create_document(
        "data/notes/spark.md",
        "Spark document",
    )

    source = normalize_source_path(document.source_path)

    detector = DocumentChangeDetector()

    changes = detector.detect(
        documents=[document],
        previous_manifest={source: create_manifest_entry(document)},
    )

    assert changes[0].status == DocumentStatus.UNCHANGED


def test_changed_document_is_detected() -> None:
    old_document = create_document(
        "data/notes/spark.md",
        "Old Spark content",
    )

    new_document = create_document(
        "data/notes/spark.md",
        "New Spark content",
    )

    source = normalize_source_path(old_document.source_path)

    detector = DocumentChangeDetector()

    changes = detector.detect(
        documents=[new_document],
        previous_manifest={source: create_manifest_entry(old_document)},
    )

    assert changes[0].status == DocumentStatus.CHANGED


def test_deleted_document_is_detected() -> None:
    document = create_document(
        "data/notes/deleted.md",
        "Deleted document",
    )

    source = normalize_source_path(document.source_path)

    detector = DocumentChangeDetector()

    changes = detector.detect(
        documents=[],
        previous_manifest={source: create_manifest_entry(document)},
    )

    assert len(changes) == 1
    assert changes[0].status == DocumentStatus.DELETED
