import json
from pathlib import Path
from typing import Any

from semantic_notes.models import ManifestEntry


class ManifestRepository:
    """
    Reads and writes indexing metadata as JSON.
    """

    def __init__(self, manifest_path: Path) -> None:
        self._manifest_path = manifest_path

    def load(self) -> dict[str, ManifestEntry]:
        if not self._manifest_path.exists():
            return {}

        raw_content = self._manifest_path.read_text(encoding="utf-8").strip()

        if not raw_content:
            return {}

        raw_data: dict[str, Any] = json.loads(raw_content)

        entries: dict[str, ManifestEntry] = {}

        for source, value in raw_data.items():
            entries[source] = ManifestEntry(
                source=source,
                document_id=str(value["document_id"]),
                content_hash=str(value["content_hash"]),
                chunk_count=int(value["chunk_count"]),
                index_signature=str(
                    value.get("index_signature", "")
                ),
            )

        return entries

    def save(
        self,
        entries: dict[str, ManifestEntry],
    ) -> None:
        self._manifest_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        serialized_entries = {
            source: {
                "document_id": entry.document_id,
                "content_hash": entry.content_hash,
                "index_signature": entry.index_signature,
                "chunk_count": entry.chunk_count,
            }
            for source, entry in sorted(entries.items())
        }

        temporary_path = self._manifest_path.with_suffix(".tmp")

        temporary_path.write_text(
            json.dumps(
                serialized_entries,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(self._manifest_path)
