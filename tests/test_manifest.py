from pathlib import Path

from semantic_notes.ingestion.manifest import (
    ManifestRepository,
)
from semantic_notes.models import ManifestEntry


def test_missing_manifest_returns_empty_dictionary(
    tmp_path: Path,
) -> None:
    repository = ManifestRepository(tmp_path / "manifest.json")

    manifest = repository.load()

    assert manifest == {}


def test_save_and_load_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"

    repository = ManifestRepository(manifest_path)

    entry = ManifestEntry(
        source="data/notes/spark.md",
        document_id="document-1",
        content_hash="content-hash-1",
        index_signature="test-index-signature",
        chunk_count=3,
    )

    repository.save(
        {
            entry.source: entry,
        }
    )

    loaded_manifest = repository.load()

    assert loaded_manifest == {
        entry.source: entry,
    }
