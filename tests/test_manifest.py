from pathlib import Path

from semantic_notes.ingestion.manifest import (
    ManifestRepository,
)
from semantic_notes.models import ManifestEntry


def test_missing_manifest_returns_empty_dictionary(
    tmp_path: Path,
) -> None:
    repository = ManifestRepository(tmp_path / "manifest.json")

    result = repository.load()

    assert result == {}


def test_save_and_load_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"

    repository = ManifestRepository(manifest_path)

    entries = {
        "data/notes/spark.md": ManifestEntry(
            source="data/notes/spark.md",
            document_id="document-123",
            content_hash="hash-123",
            chunk_count=2,
        )
    }

    repository.save(entries)

    loaded_entries = repository.load()

    assert loaded_entries == entries
