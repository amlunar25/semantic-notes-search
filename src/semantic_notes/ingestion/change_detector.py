from semantic_notes.ingestion.fingerprint import (
    calculate_content_hash,
)
from semantic_notes.ingestion.paths import normalize_source_path
from semantic_notes.models import (
    Document,
    DocumentChange,
    DocumentStatus,
    ManifestEntry,
)


class DocumentChangeDetector:
    """
    Compares current documents with the previous manifest.
    """

    def detect(
        self,
        documents: list[Document],
        previous_manifest: dict[str, ManifestEntry],
        current_index_signature: str,
    ) -> list[DocumentChange]:
        changes: list[DocumentChange] = []
        current_sources: set[str] = set()

        for document in documents:
            source = normalize_source_path(document.source_path)

            current_sources.add(source)

            current_hash = calculate_content_hash(document)
            previous_entry = previous_manifest.get(source)

            if previous_entry is None:
                changes.append(
                    DocumentChange(
                        source=source,
                        status=DocumentStatus.NEW,
                    )
                )
                continue

            content_changed = previous_entry.content_hash != current_hash

            index_configuration_changed = previous_entry.index_signature != current_index_signature

            if content_changed or index_configuration_changed:
                changes.append(
                    DocumentChange(
                        source=source,
                        status=DocumentStatus.CHANGED,
                        previous_entry=previous_entry,
                    )
                )
                continue

            changes.append(
                DocumentChange(
                    source=source,
                    status=DocumentStatus.UNCHANGED,
                    previous_entry=previous_entry,
                )
            )

        previous_sources = set(previous_manifest)

        deleted_sources = previous_sources - current_sources

        for source in sorted(deleted_sources):
            changes.append(
                DocumentChange(
                    source=source,
                    status=DocumentStatus.DELETED,
                    previous_entry=previous_manifest[source],
                )
            )

        return changes
