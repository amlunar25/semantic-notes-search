from pathlib import Path

from semantic_notes.ingestion.change_detector import (
    DocumentChangeDetector,
)
from semantic_notes.ingestion.fingerprint import (
    calculate_content_hash,
)
from semantic_notes.models import (
    Document,
    DocumentStatus,
    ManifestEntry,
)


INDEX_SIGNATURE = "test-index-signature"


def create_document(
    content: str = "Spark checkpoint information.",
) -> Document:
    return Document(
        source_path=Path("data/notes/spark.md"),
        title="Spark",
        content=content,
    )


def create_manifest_entry(
    document: Document,
    *,
    content_hash: str | None = None,
    index_signature: str = INDEX_SIGNATURE,
) -> ManifestEntry:
    return ManifestEntry(
        source=document.source_path.as_posix(),
        document_id="document-1",
        content_hash=(
            content_hash if content_hash is not None else calculate_content_hash(document)
        ),
        index_signature=index_signature,
        chunk_count=1,
    )


def test_new_document_is_detected() -> None:
    document = create_document()
    detector = DocumentChangeDetector()

    changes = detector.detect(
        documents=[document],
        previous_manifest={},
        current_index_signature=INDEX_SIGNATURE,
    )

    assert len(changes) == 1
    assert changes[0].source == "data/notes/spark.md"
    assert changes[0].status == DocumentStatus.NEW
    assert changes[0].previous_entry is None


def test_unchanged_document_is_detected() -> None:
    document = create_document()

    previous_entry = create_manifest_entry(document)

    detector = DocumentChangeDetector()

    changes = detector.detect(
        documents=[document],
        previous_manifest={
            previous_entry.source: previous_entry,
        },
        current_index_signature=INDEX_SIGNATURE,
    )

    assert len(changes) == 1
    assert changes[0].status == DocumentStatus.UNCHANGED
    assert changes[0].previous_entry == previous_entry


def test_changed_document_is_detected() -> None:
    document = create_document(content="Updated Spark checkpoint information.")

    previous_entry = create_manifest_entry(
        document,
        content_hash="old-content-hash",
    )

    detector = DocumentChangeDetector()

    changes = detector.detect(
        documents=[document],
        previous_manifest={
            previous_entry.source: previous_entry,
        },
        current_index_signature=INDEX_SIGNATURE,
    )

    assert len(changes) == 1
    assert changes[0].status == DocumentStatus.CHANGED
    assert changes[0].previous_entry == previous_entry


def test_deleted_document_is_detected() -> None:
    previous_entry = ManifestEntry(
        source="data/notes/deleted.md",
        document_id="deleted-document",
        content_hash="deleted-content-hash",
        index_signature=INDEX_SIGNATURE,
        chunk_count=1,
    )

    detector = DocumentChangeDetector()

    changes = detector.detect(
        documents=[],
        previous_manifest={
            previous_entry.source: previous_entry,
        },
        current_index_signature=INDEX_SIGNATURE,
    )

    assert len(changes) == 1
    assert changes[0].source == "data/notes/deleted.md"
    assert changes[0].status == DocumentStatus.DELETED
    assert changes[0].previous_entry == previous_entry


def test_changed_index_signature_marks_document_as_changed() -> None:
    document = create_document()

    previous_entry = create_manifest_entry(
        document,
        index_signature="old-index-signature",
    )

    detector = DocumentChangeDetector()

    changes = detector.detect(
        documents=[document],
        previous_manifest={
            previous_entry.source: previous_entry,
        },
        current_index_signature="new-index-signature",
    )

    assert len(changes) == 1
    assert changes[0].status == DocumentStatus.CHANGED
